export AIMET_VARIANT="torch_gpu"
export RELEASE_TAG="1.24.0"
export DOWNLOAD_URL="https://github.com/quic/aimet/releases/download/${RELEASE_TAG}"
export WHEEL_FILE_SUFFIX="cp38-cp38-linux_x86_64.whl"

#https://github.com/quic/aimet/releases/download/1.19.1.py37/Aimet-torch_gpu_1.19.1.py37-cp37-cp37m-linux_x86_64.whl
python3 -m pip install --upgrade pip
python3 -m pip install ${DOWNLOAD_URL}/AimetCommon-${AIMET_VARIANT}_${RELEASE_TAG}-${WHEEL_FILE_SUFFIX}
python3 -m pip install ${DOWNLOAD_URL}/AimetTorch-${AIMET_VARIANT}_${RELEASE_TAG}-${WHEEL_FILE_SUFFIX} -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install ${DOWNLOAD_URL}/Aimet-${AIMET_VARIANT}_${RELEASE_TAG}-${WHEEL_FILE_SUFFIX}