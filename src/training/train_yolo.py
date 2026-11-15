from .ultralytics_train import train

if __name__ == '__main__':
    import sys
    print(train('yolo11s', dry_run='--dry-run' in sys.argv))
