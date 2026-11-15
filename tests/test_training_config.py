from src.training.common import read_config


def test_configs():
    for name in ('yolo11s', 'rtdetr_l', 'faster_rcnn'):
        assert read_config(name)['epochs'] > 0
        assert read_config(name)['batch'] >= 1


def test_dataset_class_mapping():
    from src.common import config
    assert len(config()['classes']) == 10
