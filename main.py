import sys
from src.utils import load_config

def run_lbph():
    from src.lbph_validate import validate_lbph
    from src.lbph_test import test_lbph
    config = load_config()
    threshold = validate_lbph(config)
    print("LBPH Threshold:", threshold)
    test_lbph(config, threshold)

def run_facenet():
    from src.facenet_train import build_embeddings
    from src.facenet_validate import validate_facenet
    from src.facenet_test import test_facenet
    config = load_config()
    centroids = build_embeddings(config["data_dir"], config["cascade_path"])
    threshold = validate_facenet(config["data_dir"], config["cascade_path"], centroids)
    print("FaceNet Threshold:", threshold)
    test_facenet(config["data_dir"], config["cascade_path"], centroids, threshold)

def run_compare():
    print("\nRun LBPH first:")
    run_lbph()
    print("\nRun FaceNet next:")
    run_facenet()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [lbph | facenet | compare | live_lbph | live_facenet | live_ensemble]")
    mode = sys.argv[1].lower()
    if mode == "lbph":
        run_lbph()
    elif mode == "facenet":
        run_facenet()
    elif mode == "compare":
        run_compare()
    elif mode == "live_lbph":
        from src.live_lbph import run_lbph_live
        run_lbph_live()
    elif mode == "live_facenet":
        from src.live_facenet import run_facenet_live
        run_facenet_live()
    elif mode == "live_ensemble":
        from src.live_ensemble import run_ensemble_live
        run_ensemble_live()
    else:
        print("Unknown mode:", mode)