log_info "Uninstalling existing Triton package"
pip uninstall triton -y

log_info "Installing Triton ${TRITON_VERSION}"
pip install "triton==${TRITON_VERSION}"
