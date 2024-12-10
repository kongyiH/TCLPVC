def get_default_config(data_name):
    if data_name in ['Fashion']:
        """The default configs."""
        return dict(
            Module=dict(
                in_dim=[784, 784],
                feature_dim=512,
                high_feature_dim=128,
            ),
            Dataset=dict(
                num_sample=10000,
                num_classes=10,
                batch_size=256,
                aligned_ratio=0.5,
                num_views=2,
                workers=8,
            ),
            training=dict(
                seed=10,
                lr=0.0003,
                weight_decay=0,

                temperature_f=0.5,
                temperature_l=1,

                mse_epochs=200,
                con_epochs=50,
                tune_epochs=50,

                knn=10,
                lambda_graph=1,
            ),
        )

    elif data_name in ['BDGP']:
        """The default configs."""
        return dict(
            Module=dict(
                in_dim=[1750, 79],
                feature_dim=512,
                high_feature_dim=128,
            ),
            Dataset=dict(
                num_sample=2500,
                num_classes=5,
                num_views=2,
                aligned_ratio=0.5,
                batch_size=256,
                workers=8,
            ),
            training=dict(
                seed=10,
                lr=0.0003,
                weight_decay=0,

                temperature_f=0.5,
                temperature_l=1,

                mse_epochs=200,
                con_epochs=10,
                tune_epochs=50,

                knn=5,
                lambda_graph=0.001,
            ),
        )

    elif data_name in ['HandWritten']:
        """The default configs."""
        return dict(
            Module=dict(
                in_dim=[240, 216],
                feature_dim=512,
                high_feature_dim=128,
            ),
            Dataset=dict(
                num_sample=2000,
                num_classes=10,
                num_views=2,
                aligned_ratio=0.5,
                batch_size=256,
                workers=8,
            ),
            training=dict(
                seed=5,
                lr=0.0003,
                weight_decay=0,

                temperature_f=0.5,
                temperature_l=1,

                mse_epochs=200,
                con_epochs=200,
                tune_epochs=50,

                knn=30,
                lambda_graph=100,
            ),
        )

    elif data_name in ['Reuters_dim10']:
        return dict(
            Module=dict(
                in_dim=[10, 10],
                feature_dim=512,
                high_feature_dim=128,
            ),
            Dataset=dict(
                num_sample=18758,
                num_classes=6,
                num_views=2,
                aligned_ratio=0.5,
                batch_size=256,
                workers=8,
            ),
            training=dict(
                seed=10,
                lr=0.0003,
                weight_decay=0,

                temperature_f=0.5,
                temperature_l=1,

                mse_epochs=200,
                con_epochs=50,
                tune_epochs=50,

                knn=10,
                lambda_graph=1,
            ),
        )

    elif data_name in ['WebKB']:
        """The default configs."""
        return dict(
            Module=dict(
                in_dim=[2949, 334],
                feature_dim=512,
                high_feature_dim=128,
            ),
            Dataset=dict(
                num_sample=1051,
                num_classes=2,
                num_views=2,
                aligned_ratio=0.5,
                batch_size=32,
                workers=8,
            ),
            training=dict(
                seed=10,
                lr=0.0003,
                weight_decay=0,

                temperature_f=0.5,
                temperature_l=1,

                mse_epochs=200,
                con_epochs=80,
                tune_epochs=50,

                knn=10,
                lambda_graph=1000,
            ),
        )

    elif data_name in ['Caltech101-7']:
        return dict(
            Module=dict(
                in_dim=[1984, 512],
                feature_dim=512,
                high_feature_dim=128,
            ),
            Dataset=dict(
                num_sample=1474,
                num_classes=7,
                num_views=2,
                aligned_ratio=0.5,
                batch_size=256,
                workers=8,
            ),
            training=dict(
                seed=10,
                lr=0.0003,
                weight_decay=0,

                temperature_f=0.5,
                temperature_l=1,

                mse_epochs=200,
                con_epochs=100,
                tune_epochs=50,

                knn=8,
                lambda_graph=10000,
            ),
        )

    else:
        raise Exception('Undefined data_name')
