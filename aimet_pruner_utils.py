import os, pickle
from decimal import Decimal
import math
from typing import Tuple, List, Optional, Dict

import statistics
import torch

from aimet_common.defs import CostMetric, RankSelectScheme, EvalFunction, LayerCompRatioPair, GreedyCompressionRatioSelectionStats
from aimet_common.cost_calculator import SpatialSvdCostCalculator, WeightSvdCostCalculator
from aimet_common.comp_ratio_select import CompRatioSelectAlgo, TarRankSelectAlgo, ManualCompRatioSelectAlgo
from aimet_common.comp_ratio_rounder import RankRounder, ChannelRounder, CompRatioRounder
from aimet_common.compression_algo import CompressionAlgo
from aimet_common.bokeh_plots import BokehServerSession
from aimet_common.pruner import Pruner
from aimet_common import cost_calculator as cc
from aimet_common.utils import AimetLogger
from aimet_common.layer_database import Layer
from aimet_common.bokeh_plots import LinePlot, DataTable, LinePlot, ProgressBar
from aimet_common.curve_fit import MonotonicIncreasingCurveFit

from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
from aimet_torch.defs import SpatialSvdParameters, WeightSvdParameters, ChannelPruningParameters, ModuleCompRatioPair
from aimet_torch.layer_selector import ConvFcLayerSelector, ConvNoDepthwiseLayerSelector, ManualLayerSelector
from aimet_torch.layer_database import LayerDatabase
from aimet_torch.svd.svd_pruner import SpatialSvdPruner, WeightSvdPruner
from aimet_torch.channel_pruning.channel_pruner import InputChannelPruner, ChannelPruningCostCalculator
from aimet_torch import pymo_utils


from aimet_torch.compression_factory import CompressionFactory

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.CompRatioSelect)

class CustomInputChannelPruner(InputChannelPruner) :
    
    def _data_subsample_and_reconstruction(self, orig_layer: torch.nn.Conv2d, pruned_layer: torch.nn.Conv2d,
                                           orig_model: torch.nn.Module, comp_model: torch.nn.Module):
        #disabling with weight reconstruction step
        pass

class CustomGreedyCompRatioSelectAlgo(CompRatioSelectAlgo):
    """
    Implements the greedy compression-ratio select algorithm
    """
    # pylint: disable=too-many-locals
    def __init__(self, layer_db: LayerDatabase, pruner: Pruner, cost_calculator: cc.CostCalculator,
                 eval_func: EvalFunction, eval_iterations, cost_metric: CostMetric, target_comp_ratio: float,
                 num_candidates: int, use_monotonic_fit: bool, saved_eval_scores_dict: Optional[str],
                 comp_ratio_rounding_algo: CompRatioRounder, use_cuda: bool, bokeh_session, min_comp_ratio):

        # pylint: disable=too-many-arguments
        CompRatioSelectAlgo.__init__(self, layer_db, cost_calculator, cost_metric, comp_ratio_rounding_algo)

        self._eval_func = eval_func
        self.bokeh_session = bokeh_session
        self._eval_iter = eval_iterations
        self._is_cuda = use_cuda
        self._pruner = pruner
        self._saved_eval_scores_dict = saved_eval_scores_dict
        self._target_comp_ratio = target_comp_ratio
        self._use_monotonic_fit = use_monotonic_fit

        if saved_eval_scores_dict:
            self._comp_ratio_candidates = 0

        else:

            ratios = []
            step = (1 - min_comp_ratio)/num_candidates
            for i in range(num_candidates) :
                ratios.append(Decimal(min_comp_ratio + step*i))
            self._comp_ratio_candidates = ratios
            self.target_ratios = ratios

        CompRatioSelectAlgo.__init__(self, layer_db, cost_calculator, cost_metric, comp_ratio_rounding_algo)

    def _pickle_eval_scores_dict(self, eval_scores_dict):

        if not os.path.exists('./data'):
            os.makedirs('./data')

        with open(self.PICKLE_FILE_EVAL_DICT, 'wb') as file:
            pickle.dump(eval_scores_dict, file)

        logger.info("Greedy selection: Saved eval dict to %s", self.PICKLE_FILE_EVAL_DICT)

    @staticmethod
    def _unpickle_eval_scores_dict(saved_eval_scores_dict_path: str):

        with open(saved_eval_scores_dict_path, 'rb') as f:
            eval_dict = pickle.load(f)

        logger.info("Greedy selection: Read eval dict from %s", saved_eval_scores_dict_path)
        return eval_dict

    @staticmethod
    def _calculate_function_value_by_interpolation(comp_ratio: Decimal, layer_eval_score_dict: dict,
                                                   comp_ratio_list: List):
        """
        Calculates eval score for a comp ratio by interpolation
        :param comp_ratio:
        :param layer_eval_score_dict:
        :param comp_ratio_list:
        :return:
        """
        if comp_ratio in comp_ratio_list:
            eval_score = layer_eval_score_dict[comp_ratio]
        else:
            ind = 0
            for ind, _ in enumerate(comp_ratio_list, start=0):
                if comp_ratio < comp_ratio_list[ind]:
                    break

            if ind == len(comp_ratio_list) - 1:
                eval_score = layer_eval_score_dict[comp_ratio_list[-1]]
            else:
                x1 = comp_ratio_list[ind]
                y1 = layer_eval_score_dict[comp_ratio_list[ind]]
                x2 = comp_ratio_list[ind - 1]
                y2 = layer_eval_score_dict[comp_ratio_list[ind - 1]]
                eval_score = (float(comp_ratio) - float(x1)) * (y1 - y2) / (float(x1) - float(x2)) + y1
        return eval_score

    def _update_eval_dict_with_rounding(self, eval_scores_dict, rounding_algo, cost_metric):
        updated_eval_dict = {}

        for layer_name in eval_scores_dict:
            layer_eval_dict = eval_scores_dict[layer_name]
            eval_dict_per_layer = {}

            layer = self._layer_db.find_layer_by_name(layer_name)
            comp_ratio_list = sorted(list(layer_eval_dict.keys()), key=float)
            for comp_ratio in layer_eval_dict:
                rounded_comp_ratio = rounding_algo.round(layer, comp_ratio, cost_metric)

                eval_score = self._calculate_function_value_by_interpolation(rounded_comp_ratio, layer_eval_dict,
                                                                             comp_ratio_list)
                eval_dict_per_layer[Decimal(rounded_comp_ratio)] = eval_score
            updated_eval_dict[layer_name] = eval_dict_per_layer
        return updated_eval_dict

    @staticmethod
    def _fit_eval_dict_to_monotonic_function(eval_scores_dict):

        for layer in eval_scores_dict:
            layer_eval_dict = eval_scores_dict[layer]
            # Convert dict of eval-scores and comp-ratios to lists
            eval_scores = list(layer_eval_dict.values())
            comp_ratios = list(layer_eval_dict.keys())

            eval_scores, polynomial_coefficients = MonotonicIncreasingCurveFit.fit(comp_ratios, eval_scores)
            logger.debug("The coefficients for layer %s are %s", layer, str(polynomial_coefficients))
            # Update the layer_eval_dict
            for index, comp_ratio in enumerate(comp_ratios):
                layer_eval_dict[comp_ratio] = eval_scores[index]

    def _construct_eval_dict(self):
        #  If the user already passed in a previously saved eval scores dict, we just use that
        if self._saved_eval_scores_dict:
            eval_scores_dict = self._unpickle_eval_scores_dict(self._saved_eval_scores_dict)

        else:
            # Create the eval scores dictionary
            eval_scores_dict = self._compute_eval_scores_for_all_comp_ratio_candidates()

            # save the dictionary to file (in case the user wants to reuse the dictionary in the future)
            self._pickle_eval_scores_dict(eval_scores_dict)
        return eval_scores_dict

    def select_per_layer_comp_ratios(self):

        # Compute eval scores for each candidate comp-ratio in each layer
        eval_scores_dict = self._construct_eval_dict()

        # Fit the scores to a monotonically increasing function
        if self._use_monotonic_fit:
            self._fit_eval_dict_to_monotonic_function(eval_scores_dict)

        updated_eval_scores_dict = self._update_eval_dict_with_rounding(eval_scores_dict, self._rounding_algo,
                                                                        self._cost_metric)

        # Get the overall min and max scores
        current_min_score, current_max_score = self._find_min_max_eval_scores(updated_eval_scores_dict)
        exit_threshold = (current_max_score - current_min_score) * 0.0001
        logger.info("Greedy selection: overall_min_score=%f, overall_max_score=%f",
                    current_min_score, current_max_score)

        # Base cost
        original_model_cost = self._cost_calculator.compute_model_cost(self._layer_db)
        logger.info("Greedy selection: Original model cost=%s", original_model_cost)

        while True:

            # Current mid-point score
            current_mid_score = statistics.mean([current_max_score, current_min_score])
            current_comp_ratio = self._calculate_model_comp_ratio_for_given_eval_score(current_mid_score,
                                                                                       updated_eval_scores_dict,
                                                                                       original_model_cost)

            logger.debug("Greedy selection: current candidate - comp_ratio=%f, score=%f, search-window=[%f:%f]",
                         current_comp_ratio, current_mid_score, current_min_score, current_max_score)

            # Exit condition: is the binary search window too small to continue?
            should_exit, selected_score = self._evaluate_exit_condition(current_min_score, current_max_score,
                                                                        exit_threshold,
                                                                        current_comp_ratio, self._target_comp_ratio)

            if should_exit:
                break

            if current_comp_ratio > self._target_comp_ratio:
                # Not enough compression: Binary search the lower half of the scores
                current_max_score = current_mid_score
            else:
                # Too much compression: Binary search the upper half of the scores
                current_min_score = current_mid_score

        # Search finished, return the selected comp ratios per layer
        # Calculate the compression ratios for each layer based on this score
        layer_ratio_list = self._find_all_comp_ratios_given_eval_score(selected_score, updated_eval_scores_dict)
        selected_comp_ratio = self._calculate_model_comp_ratio_for_given_eval_score(selected_score,
                                                                                    updated_eval_scores_dict,
                                                                                    original_model_cost)

        logger.info("Greedy selection: final choice - comp_ratio=%f, score=%f",
                    selected_comp_ratio, selected_score)

        return layer_ratio_list, GreedyCompressionRatioSelectionStats(updated_eval_scores_dict)

    @staticmethod
    def _evaluate_exit_condition(min_score, max_score, exit_threshold, current_comp_ratio, target_comp_ratio):

        if math.isclose(min_score, max_score, abs_tol=exit_threshold):
            return True, min_score

        if math.isclose(current_comp_ratio, target_comp_ratio, abs_tol=0.001):
            return True, statistics.mean([max_score, min_score])

        return False, None

    def _calculate_model_comp_ratio_for_given_eval_score(self, eval_score, eval_scores_dict,
                                                         original_model_cost):

        # Calculate the compression ratios for each layer based on this score
        layer_ratio_list = self._find_all_comp_ratios_given_eval_score(eval_score, eval_scores_dict)
        for layer in self._layer_db:
            if layer not in self._layer_db.get_selected_layers():
                layer_ratio_list.append(LayerCompRatioPair(layer, None))

        # Calculate compressed model cost
        compressed_model_cost = self._cost_calculator.calculate_compressed_cost(self._layer_db,
                                                                                layer_ratio_list,
                                                                                self._cost_metric)

        if self._cost_metric == CostMetric.memory:
            current_comp_ratio = Decimal(compressed_model_cost.memory / original_model_cost.memory)
        else:
            current_comp_ratio = Decimal(compressed_model_cost.mac / original_model_cost.mac)

        return current_comp_ratio

    def _find_all_comp_ratios_given_eval_score(self, given_eval_score, eval_scores_dict):
        layer_ratio_list = []
        for layer in self._layer_db.get_selected_layers():
            comp_ratio = self._find_layer_comp_ratio_given_eval_score(eval_scores_dict,
                                                                      given_eval_score, layer)
            layer_ratio_list.append(LayerCompRatioPair(layer, comp_ratio))

        return layer_ratio_list

    @staticmethod
    def _find_layer_comp_ratio_given_eval_score(eval_scores_dict: Dict[str, Dict[Decimal, float]],
                                                given_eval_score, layer: Layer):

        # Find the closest comp ratio candidate for the current eval score
        eval_scores_for_layer = eval_scores_dict[layer.name]

        # Sort the eval scores by increasing order of compression
        comp_ratios = list(eval_scores_for_layer.keys())
        sorted_comp_ratios = sorted(comp_ratios, reverse=True)

        # Special cases
        # Case1: Eval score is higher than even our most conservative comp ratio: then no compression
        if given_eval_score > eval_scores_for_layer[sorted_comp_ratios[0]]:
            return None

        if given_eval_score < eval_scores_for_layer[sorted_comp_ratios[-1]]:
            return sorted_comp_ratios[-1]

        # Start with a default of no compression
        selected_comp_ratio = None

        for index, comp_ratio in enumerate(sorted_comp_ratios[1:]):

            if given_eval_score > eval_scores_for_layer[comp_ratio]:
                selected_comp_ratio = sorted_comp_ratios[index]
                break

        return selected_comp_ratio

    @staticmethod
    def _find_min_max_eval_scores(eval_scores_dict: Dict[str, Dict[Decimal, float]]):
        first_layer_scores = list(eval_scores_dict.values())[0]
        first_score = list(first_layer_scores.values())[0]

        min_score = first_score
        max_score = first_score

        for layer_scores in eval_scores_dict.values():
            for eval_score in layer_scores.values():

                if eval_score < min_score:
                    min_score = eval_score

                if eval_score > max_score:
                    max_score = eval_score

        return min_score, max_score

    def _compute_eval_scores_for_all_comp_ratio_candidates(self) -> Dict[str, Dict[Decimal, float]]:
        """
        Creates and returns the eval scores dictionary
        :return: Dictionary of {layer_name: {compression_ratio: eval_score}}  for all selected layers
                 and all compression-ratio candidates
        """

        selected_layers = self._layer_db.get_selected_layers()

        # inputs to initialize a TabularProgress object
        num_candidates = len(self._comp_ratio_candidates)
        num_layers = len(selected_layers)

        if self.bokeh_session:
            column_names = [str(i) for i in self._comp_ratio_candidates]
            layer_names = [i.name for i in selected_layers]

            progress_bar = ProgressBar(total=num_layers * num_candidates, title="Eval Scores Table", color="green",
                                       bokeh_session=self.bokeh_session)
            data_table = DataTable(num_layers, num_candidates, column_names, bokeh_session=self.bokeh_session,
                                   row_index_names=layer_names)
        else:
            data_table = None
            progress_bar = None

        eval_scores_dict = {}
        for layer in selected_layers:

            layer_wise_eval_scores = self._compute_layerwise_eval_score_per_comp_ratio_candidate(data_table,
                                                                                                 progress_bar, layer)
            eval_scores_dict[layer.name] = layer_wise_eval_scores

        return eval_scores_dict

    def _compute_layerwise_eval_score_per_comp_ratio_candidate(self, tabular_progress_object, progress_bar,
                                                               layer: Layer) -> Dict[Decimal, float]:
        """
        Computes eval scores for each compression-ratio candidate for a given layer
        :param layer: Layer for which to calculate eval scores
        :return: Dictionary of {compression_ratio: eval_score} for each compression-ratio candidate
        """

        layer_wise_eval_scores_dict = {}

        # Only publish plots to a document if a bokeh server session exists
        if self.bokeh_session:

            # plot to visualize the evaluation scores as they update for each layer
            layer_wise_eval_scores_plot = LinePlot(x_axis_label="Compression Ratios", y_axis_label="Eval Scores",
                                                   title=layer.name, bokeh_session=self.bokeh_session)
        # Loop over each candidate
        #logger.info("Candidate Ratios",self.target_ratios)
        for comp_ratio in self.target_ratios:
            logger.info("Analyzing compression ratio: %s =====================>", comp_ratio)

            # Prune layer given this comp ratio
            pruned_layer_db = self._pruner.prune_model(self._layer_db,
                                                       [LayerCompRatioPair(layer, comp_ratio)],
                                                       self._cost_metric,
                                                       trainer=None)

            eval_score = self._eval_func(pruned_layer_db.model, self._eval_iter, use_cuda=self._is_cuda)
            layer_wise_eval_scores_dict[comp_ratio] = eval_score

            # destroy the layer database
            pruned_layer_db.destroy()
            pruned_layer_db = None

            logger.info("Layer %s, comp_ratio %f ==> eval_score=%f", layer.name, comp_ratio,
                        eval_score)

            if self.bokeh_session:
                layer_wise_eval_scores_plot.update(new_x_coordinate=comp_ratio, new_y_coordinate=eval_score)
                # Update the data table by adding the computed eval score
                tabular_progress_object.update_table(str(comp_ratio), layer.name, eval_score)
                # Update the progress bar
                progress_bar.update()

        # remove plot so that we have a fresh figure to visualize for the next layer.
        if self.bokeh_session:
            layer_wise_eval_scores_plot.remove_plot()

        return layer_wise_eval_scores_dict


class CustomCompressionFactory(CompressionFactory) :
    @classmethod
    def create_channel_pruning_algo(
            cls, 
            model: torch.nn.Module, 
            eval_callback: EvalFunction, 
            eval_iterations,
            input_shape: Tuple, cost_metric: CostMetric,
            params: ChannelPruningParameters,
            bokeh_session: BokehServerSession,
            min_comp_ratio:float = 0,  
            ) -> CompressionAlgo:
            """
            Factory method to construct ChannelPruningCompressionAlgo
            :param model: Model to compress
            :param eval_callback: Evaluation callback for the model
            :param eval_iterations: Evaluation iterations
            :param input_shape: Shape of the input tensor for model
            :param cost_metric: Cost metric (mac or memory)
            :param params: Channel Pruning compression parameters
            :param bokeh_session: The Bokeh session to display plots
            :return: An instance of ChannelPruningCompressionAlgo
            """
    
            # pylint: disable=too-many-locals
            # Rationale: Factory functions unfortunately need to deal with a lot of parameters
    
            device = get_device(model)
            dummy_input = create_rand_tensors_given_shapes(input_shape, device)
    
            # Create a layer database
            layer_db = LayerDatabase(model, dummy_input)
            use_cuda = next(model.parameters()).is_cuda
    
            # Create a pruner
            pruner = CustomInputChannelPruner(data_loader=params.data_loader, input_shape=input_shape,
                                        num_reconstruction_samples=params.num_reconstruction_samples,
                                        allow_custom_downsample_ops=params.allow_custom_downsample_ops)
            comp_ratio_rounding_algo = ChannelRounder(params.multiplicity)
    
            # Create a comp-ratio selection algorithm
            cost_calculator = ChannelPruningCostCalculator(pruner)
    
            if params.mode == ChannelPruningParameters.Mode.auto:
                greedy_params = params.mode_params.greedy_params
                comp_ratio_select_algo = CustomGreedyCompRatioSelectAlgo(layer_db, pruner, cost_calculator, eval_callback,
                                                                   eval_iterations, cost_metric,
                                                                   greedy_params.target_comp_ratio,
                                                                   greedy_params.num_comp_ratio_candidates,
                                                                   greedy_params.use_monotonic_fit,
                                                                   greedy_params.saved_eval_scores_dict,
                                                                   comp_ratio_rounding_algo, use_cuda,
                                                                   bokeh_session=bokeh_session,
                                                                   min_comp_ratio=min_comp_ratio
                                                                   )
    
                layer_selector = ConvNoDepthwiseLayerSelector()
                modules_to_ignore = params.mode_params.modules_to_ignore
    
            else:
                # Convert (module,comp-ratio) pairs to (layer,comp-ratio) pairs
                layer_comp_ratio_pairs = cls._get_layer_pairs(layer_db, params.mode_params.list_of_module_comp_ratio_pairs)
    
                comp_ratio_select_algo = ManualCompRatioSelectAlgo(layer_db,
                                                                   layer_comp_ratio_pairs,
                                                                   comp_ratio_rounding_algo, cost_metric=cost_metric)
    
                layer_selector = ManualLayerSelector(layer_comp_ratio_pairs)
                modules_to_ignore = []
    
            # Create the overall Channel Pruning compression algorithm
            channel_pruning_algo = CompressionAlgo(layer_db, comp_ratio_select_algo, pruner, eval_callback,
                                                   layer_selector, modules_to_ignore, cost_calculator, use_cuda)
            print("internal:", channel_pruning_algo._comp_ratio_select_algo)
            return channel_pruning_algo