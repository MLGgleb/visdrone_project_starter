from .ultralytics_train import train

if __name__ == '__main__':
    import sys
    print(train('rtdetr_l', dry_run='--dry-run' in sys.argv))
