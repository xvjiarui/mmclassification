_base_ = './resnet50_batch256.py'
model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 4),
            stages=(True, True, True, True),
            position='after_conv3')
    ]))
