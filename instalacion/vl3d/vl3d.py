"""
:author: Alberto M. Esmoris Pena

The main entry point for the execution of the VL3D software.
"""


# ---   MAIN   --- #
# ---------------- #
if __name__ == '__main__':
    # Disable tensorflow messages
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load logging system
    import src.main.main_logger as LOGGING
    vl3d_root = os.path.dirname(__file__)
    LOGGING.main_logger_init(rootdir=vl3d_root)

    # Load C++ extensions
    from src.vl3dpp import vl3dpp_loader
    vl3dpp_loader.vl3dpp_load(logging=False, warning=True)

    # Call main
    from src.main.main import main
    main(rootdir=vl3d_root)
