.. _example_hessig_kpconv:

KPConv-like model for point-wise classification in the Hessigheim March2018 dataset
**************************************************************************************

Data
=======

The point clouds used in this example are available in the
`official webpage of the dataset <https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/subscribe.aspx>`_.
As this example is part of a more thorough analysis, the point clouds were
preprocessed to transform the color components from RGB to HSV (see
:ref:`HSV from RGB miner documentation <HSV from RGB miner>`). The JSONs
are available in *spec/demo/kpconv_hessig_2018* with file names
*hsv_std_mine_start_pipeline.json* and *hsv_std_mine_continue_pipeline.json*,
respectively. Note, however, that only the :math:`(x, y, z)` coordinates are
considered in this example, i.e., the color components are not considered by
the model.


JSON
========

Apart from the two previously mentioned JSON files to preprocess the data, four
JSON files are introduced here. Two are used for training the model in two
consecutive training processes, and two are used to compute predictions with
the model and evaluate them. All the JSONs to reproduce this example can be
found in the *spec/demo/kpconv_hessig_2018* folder of the VL3D++ framework.

First training process JSON
-------------------------------

The JSON below can be used to training a
:ref:`KPConv-like model <Hierarchical KPConv>`
for the first time
on the training point cloud of the March2018 Hessigheim dataset. Note that the
paths to the files might need an update to fit your local file system.

.. code-block:: json

    {
      "in_pcloud": [
        "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/mined/Mar18_train_hsv_std.laz"
      ],
      "out_pcloud": [
        "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/out/kpc_ATRY7/T1/*"
      ],
      "sequential_pipeline": [
        {
            "train": "ConvolutionalAutoencoderPwiseClassifier",
            "training_type": "base",
            "fnames": ["ones"],
            "random_seed": null,
            "model_args": {
                "fnames": ["ones"],
                "num_classes": 11,
                "class_names": ["LowVeg", "ImpSurf", "Vehicle", "UrbanFurni", "Roof", "Facade", "Shrub", "Tree", "Soil/Gravel", "VertSurf", "Chimney"],
                "pre_processing": {
                    "pre_processor": "hierarchical_fps",
                    "support_strategy_num_points": 45000,
                    "to_unit_sphere": false,
                    "support_strategy": "fps",
                    "support_chunk_size": 5000,
                    "support_strategy_fast": true,
                    "center_on_pcloud": true,
                    "_training_class_distribution": [32, 0, 32, 0, 32, 32, 0, 32, 0, 0, 32],
                    "neighborhood": {
                        "type": "sphere",
                        "radius": 3.0,
                        "separation_factor": 0.8
                    },
                    "num_points_per_depth": [512, 256, 128, 64, 32],
                    "fast_flag_per_depth": [false, false, false, false, false],
                    "num_downsampling_neighbors": [1, 16, 16, 16, 16],
                    "num_pwise_neighbors": [16, 16, 16, 16, 16],
                    "num_upsampling_neighbors": [1, 16, 16, 16, 16],
                    "nthreads": 12,
                    "training_receptive_fields_distribution_report_path": "*/training_eval/training_receptive_fields_distribution.log",
                    "_training_receptive_fields_distribution_report_path": null,
                    "training_receptive_fields_distribution_plot_path": "*/training_eval/training_receptive_fields_distribution.svg",
                    "_training_receptive_fields_distribution_plot_path": null,
                    "training_receptive_fields_dir": "*/training_eval/training_receptive_fields/",
                    "training_receptive_fields_dir": null,
                    "receptive_fields_distribution_report_path": "*/training_eval/receptive_fields_distribution.log",
                    "_receptive_fields_distribution_report_path": null,
                    "receptive_fields_distribution_plot_path": "*/training_eval/receptive_fields_distribution.svg",
                    "_receptive_fields_distribution_plot_path": null,
                    "_receptive_fields_dir": "*/training_eval/receptive_fields/",
                    "receptive_fields_dir": null,
                    "training_support_points_report_path": "*/training_eval/training_support_points.laz",
                    "support_points_report_path": "*/training_eval/support_points.laz"
                },
                "feature_extraction": {
                    "type": "KPConv",
                    "operations_per_depth": [2, 1, 1, 1, 1],
                    "feature_space_dims": [64, 64, 128, 256, 512, 1024],
                    "bn": true,
                    "bn_momentum": 0.98,
                    "activate": true,
                    "sigma": [3.0, 3.0, 3.75, 4.5, 5.25, 6.0],
                    "kernel_radius": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                    "num_kernel_points": [15, 15, 15, 15, 15, 15],
                    "deformable": [false, false, false, false, false, false],
                    "W_initializer": ["glorot_uniform", "glorot_uniform", "glorot_uniform", "glorot_uniform", "glorot_uniform", "glorot_uniform"],
                    "W_regularizer": [null, null, null, null, null, null],
                    "W_constraint": [null, null, null, null, null, null],
                    "unary_convolution_wrapper": {
                        "activation": "relu",
                        "initializer": "glorot_uniform",
                        "bn": true,
                        "bn_momentum": 0.98
                    }
                },
                "_structure_alignment": {
                    "tnet_pre_filters_spec": [64, 128, 256],
                    "tnet_post_filters_spec": [128, 64, 32],
                    "kernel_initializer": "glorot_normal"
                },
                "features_alignment": null,
                "downsampling_filter": "strided_kpconv",
                "upsampling_filter": "mean",
                "upsampling_bn": true,
                "upsampling_momentum": 0.98,
                "conv1d_kernel_initializer": "glorot_normal",
                "output_kernel_initializer": "glorot_normal",
                "model_handling": {
                    "summary_report_path": "*/model_summary.log",
                    "training_history_dir": "*/training_eval/history",
                    "_features_structuring_representation_dir": "*/training_eval/feat_struct_layer/",
                    "kpconv_representation_dir": "*/training_eval/kpconv_layers/",
                    "skpconv_representation_dir": "*/training_eval/skpconv_layers/",
                    "class_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "training_epochs": 300,
                    "batch_size": 16,
                    "training_sequencer": {
                        "type": "DLSequencer",
                        "random_shuffle_indices": true,
                        "augmentor": {
                            "transformations": [
                                    {
                                        "type": "Rotation",
                                        "axis": [0, 0, 1],
                                        "angle_distribution": {
                                            "type": "uniform",
                                            "start": -3.141592,
                                            "end": 3.141592
                                        }
                                    },
                                    {
                                        "type": "Scale",
                                        "scale_distribution": {
                                            "type": "uniform",
                                            "start": 0.99,
                                            "end": 1.01
                                        }
                                    },
                                    {
                                        "type": "Jitter",
                                        "noise_distribution": {
                                            "type": "normal",
                                            "mean": 0,
                                            "stdev": 0.001
                                        }
                                    }
                            ]
                        }
                    },
                    "prediction_reducer": {
                        "reduce_strategy" : {
                            "type": "MeanPredReduceStrategy"
                        },
                        "select_strategy": {
                            "type": "ArgMaxPredSelectStrategy"
                        }
                    },
                    "checkpoint_path": "*/checkpoint.model.keras",
                    "checkpoint_monitor": "loss",
                    "learning_rate_on_plateau": {
                        "monitor": "loss",
                        "mode": "min",
                        "factor": 0.1,
                        "patience": 2000,
                        "cooldown": 5,
                        "min_delta": 0.01,
                        "min_lr": 1e-6
                    }
                },
                "compilation_args": {
                    "optimizer": {
                        "algorithm": "Adam",
                        "learning_rate": {
                            "schedule": "exponential_decay",
                            "schedule_args": {
                                "initial_learning_rate": 1e-2,
                                "decay_steps": 10000,
                                "decay_rate": 0.96,
                                "staircase": false
                            }
                        }
                    },
                    "loss": {
                        "function": "class_weighted_categorical_crossentropy"
                    },
                    "metrics": [
                        "categorical_accuracy"
                    ]
                },
                "architecture_graph_path": "*/model_graph.png",
                "architecture_graph_args": {
                    "show_shapes": true,
                    "show_dtype": true,
                    "show_layer_names": true,
                    "rankdir": "TB",
                    "expand_nested": true,
                    "dpi": 300,
                    "show_layer_activations": true
                }
            },
            "autoval_metrics": ["OA", "P", "R", "F1", "IoU", "wP", "wR", "wF1", "wIoU", "MCC", "Kappa"],
            "training_evaluation_metrics": ["OA", "P", "R", "F1", "IoU", "wP", "wR", "wF1", "wIoU", "MCC", "Kappa"],
            "training_class_evaluation_metrics": ["P", "R", "F1", "IoU"],
            "training_evaluation_report_path": "*/training_eval/evaluation.log",
            "training_class_evaluation_report_path": "*/training_eval/class_evaluation.log",
            "training_confusion_matrix_report_path": "*/training_eval/confusion.log",
            "training_confusion_matrix_plot_path": "*/training_eval/confusion.svg",
            "training_class_distribution_report_path": "*/training_eval/class_distribution.log",
            "training_class_distribution_plot_path": "*/training_eval/class_distribution.svg",
            "training_classified_point_cloud_path": "*/training_eval/classified_point_cloud.laz",
            "_training_activations_path": "*/training_eval/activations.laz",
            "training_activations_path": null
        },
        {
          "writer": "PredictivePipelineWriter",
          "out_pipeline": "*pipe/KPC_T1.pipe",
          "include_writer": false,
          "include_imputer": true,
          "include_feature_transformer": true,
          "include_miner": true,
          "include_class_transformer": true
        }
      ]
    }



Second training process JSON
-------------------------------

After the first training process, we have a trained model. With the JSON below,
we randomly select another set of 45,000 neighborhoods to train the model one
more time. Note that the initial learning rate is smaller (:math:`10^{-3}`
instead of :math:`10^{-2}`), the steps for the learning rate decay have been
increased, and the number of epochs has been reduced. These changes in the
optimization process aim to keep what has been learnt by the model while
updating it with a different perspective from the same training point cloud.

.. code-block:: json

    {
      "in_pcloud": [
        "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/mined/Mar18_train_hsv_std.laz"
      ],
      "out_pcloud": [
        "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/out/kpc_ATRY7/T2/*"
      ],
      "sequential_pipeline": [
        {
            "train": "ConvolutionalAutoencoderPwiseClassifier",
            "training_type": "base",
            "pretrained_model": "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/out/kpc_ATRY7/T1/pipe/KPC_T1.model",
            "fnames": ["ones"],
            "random_seed": null,
            "model_args": {
                "fnames": ["ones"],
                "num_classes": 11,
                "class_names": ["LowVeg", "ImpSurf", "Vehicle", "UrbanFurni", "Roof", "Facade", "Shrub", "Tree", "Soil/Gravel", "VertSurf", "Chimney"],
                "pre_processing": {
                    "pre_processor": "hierarchical_fps",
                    "support_strategy_num_points": 45000,
                    "to_unit_sphere": false,
                    "support_strategy": "fps",
                    "support_chunk_size": 5000,
                    "support_strategy_fast": true,
                    "center_on_pcloud": true,
                    "_training_class_distribution": [32, 0, 32, 0, 32, 32, 0, 32, 0, 0, 32],
                    "neighborhood": {
                        "type": "sphere",
                        "radius": 3.0,
                        "separation_factor": 0.8
                    },
                    "num_points_per_depth": [512, 256, 128, 64, 32],
                    "fast_flag_per_depth": [false, false, false, false, false],
                    "num_downsampling_neighbors": [1, 16, 16, 16, 16],
                    "num_pwise_neighbors": [16, 16, 16, 16, 16],
                    "num_upsampling_neighbors": [1, 16, 16, 16, 16],
                    "nthreads": 12,
                    "training_receptive_fields_distribution_report_path": "*/training_eval/training_receptive_fields_distribution.log",
                    "_training_receptive_fields_distribution_report_path": null,
                    "training_receptive_fields_distribution_plot_path": "*/training_eval/training_receptive_fields_distribution.svg",
                    "_training_receptive_fields_distribution_plot_path": null,
                    "training_receptive_fields_dir": "*/training_eval/training_receptive_fields/",
                    "_training_receptive_fields_dir": null,
                    "receptive_fields_distribution_report_path": "*/training_eval/receptive_fields_distribution.log",
                    "_receptive_fields_distribution_report_path": null,
                    "receptive_fields_distribution_plot_path": "*/training_eval/receptive_fields_distribution.svg",
                    "_receptive_fields_distribution_plot_path": null,
                    "_receptive_fields_dir": "*/training_eval/receptive_fields/",
                    "receptive_fields_dir": null,
                    "training_support_points_report_path": "*/training_eval/training_support_points.laz",
                    "support_points_report_path": "*/training_eval/support_points.laz"
                },
                "feature_extraction": {
                    "type": "KPConv",
                    "operations_per_depth": [2, 1, 1, 1, 1],
                    "feature_space_dims": [64, 64, 128, 256, 512, 1024],
                    "bn": true,
                    "bn_momentum": 0.98,
                    "activate": true,
                    "sigma": [3.0, 3.0, 3.75, 4.5, 5.25, 6.0],
                    "kernel_radius": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                    "num_kernel_points": [15, 15, 15, 15, 15, 15],
                    "deformable": [false, false, false, false, false, false],
                    "W_initializer": ["glorot_uniform", "glorot_uniform", "glorot_uniform", "glorot_uniform", "glorot_uniform", "glorot_uniform"],
                    "W_regularizer": [null, null, null, null, null, null],
                    "W_constraint": [null, null, null, null, null, null],
                    "unary_convolution_wrapper": {
                        "activation": "relu",
                        "initializer": "glorot_uniform",
                        "bn": true,
                        "bn_momentum": 0.98
                    }
                },
                "_structure_alignment": {
                    "tnet_pre_filters_spec": [64, 128, 256],
                    "tnet_post_filters_spec": [128, 64, 32],
                    "kernel_initializer": "glorot_normal"
                },
                "features_alignment": null,
                "downsampling_filter": "strided_kpconv",
                "upsampling_filter": "mean",
                "upsampling_bn": true,
                "upsampling_momentum": 0.98,
                "conv1d_kernel_initializer": "glorot_normal",
                "output_kernel_initializer": "glorot_normal",
                "model_handling": {
                    "summary_report_path": "*/model_summary.log",
                    "training_history_dir": "*/training_eval/history",
                    "_features_structuring_representation_dir": "*/training_eval/feat_struct_layer/",
                    "kpconv_representation_dir": "*/training_eval/kpconv_layers/",
                    "skpconv_representation_dir": "*/training_eval/skpconv_layers/",
                    "class_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "training_epochs": 200,
                    "batch_size": 16,
                    "training_sequencer": {
                        "type": "DLSequencer",
                        "random_shuffle_indices": true,
                        "augmentor": {
                            "transformations": [
                                    {
                                        "type": "Rotation",
                                        "axis": [0, 0, 1],
                                        "angle_distribution": {
                                            "type": "uniform",
                                            "start": -3.141592,
                                            "end": 3.141592
                                        }
                                    },
                                    {
                                        "type": "Scale",
                                        "scale_distribution": {
                                            "type": "uniform",
                                            "start": 0.99,
                                            "end": 1.01
                                        }
                                    },
                                    {
                                        "type": "Jitter",
                                        "noise_distribution": {
                                            "type": "normal",
                                            "mean": 0,
                                            "stdev": 0.001
                                        }
                                    }
                            ]
                        }
                    },
                    "prediction_reducer": {
                        "reduce_strategy" : {
                            "type": "MeanPredReduceStrategy"
                        },
                        "select_strategy": {
                            "type": "ArgMaxPredSelectStrategy"
                        }
                    },
                    "checkpoint_path": "*/checkpoint.model.keras",
                    "checkpoint_monitor": "loss",
                    "learning_rate_on_plateau": {
                        "monitor": "loss",
                        "mode": "min",
                        "factor": 0.1,
                        "patience": 2000,
                        "cooldown": 5,
                        "min_delta": 0.01,
                        "min_lr": 1e-6
                    }
                },
                "compilation_args": {
                    "optimizer": {
                        "algorithm": "Adam",
                        "learning_rate": {
                            "schedule": "exponential_decay",
                            "schedule_args": {
                                "initial_learning_rate": 1e-3,
                                "decay_steps": 15000,
                                "decay_rate": 0.96,
                                "staircase": false
                            }
                        }
                    },
                    "loss": {
                        "function": "class_weighted_categorical_crossentropy"
                    },
                    "metrics": [
                        "categorical_accuracy"
                    ]
                },
                "architecture_graph_path": "*/model_graph.png",
                "architecture_graph_args": {
                    "show_shapes": true,
                    "show_dtype": true,
                    "show_layer_names": true,
                    "rankdir": "TB",
                    "expand_nested": true,
                    "dpi": 300,
                    "show_layer_activations": true
                }
            },
            "autoval_metrics": ["OA", "P", "R", "F1", "IoU", "wP", "wR", "wF1", "wIoU", "MCC", "Kappa"],
            "training_evaluation_metrics": ["OA", "P", "R", "F1", "IoU", "wP", "wR", "wF1", "wIoU", "MCC", "Kappa"],
            "training_class_evaluation_metrics": ["P", "R", "F1", "IoU"],
            "training_evaluation_report_path": "*/training_eval/evaluation.log",
            "training_class_evaluation_report_path": "*/training_eval/class_evaluation.log",
            "training_confusion_matrix_report_path": "*/training_eval/confusion.log",
            "training_confusion_matrix_plot_path": "*/training_eval/confusion.svg",
            "training_class_distribution_report_path": "*/training_eval/class_distribution.log",
            "training_class_distribution_plot_path": "*/training_eval/class_distribution.svg",
            "training_classified_point_cloud_path": "*/training_eval/classified_point_cloud.laz",
            "_training_activations_path": "*/training_eval/activations.laz",
            "training_activations_path": null
        },
        {
          "writer": "PredictivePipelineWriter",
          "out_pipeline": "*pipe/KPC_T2.pipe",
          "include_writer": false,
          "include_imputer": true,
          "include_feature_transformer": true,
          "include_miner": true,
          "include_class_transformer": true
        }
      ]
    }




Predictions after the first training process JSON
----------------------------------------------------

The JSON below can be used to classify the validation point cloud with the
model after the first training process. It also computes the evaluation of the
model and the uncertainty assessment.

.. code-block:: json

    {
      "in_pcloud": [
        "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/mined/Mar18_val_hsv_std.laz"
      ],
      "out_pcloud": [
        "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/out/kpc_ATRY7/T1/preds/*"
      ],
      "sequential_pipeline": [
        {
          "predict": "PredictivePipeline",
          "model_path": "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/out/kpc_ATRY7/T1/pipe/KPC_T1.pipe"
        },
        {
            "writer": "ClassifiedPcloudWriter",
            "out_pcloud": "*predicted.las"
        },
        {
          "writer": "PredictionsWriter",
          "out_preds": "*predictions.lbl"
        },
        {
          "eval": "ClassificationEvaluator",
          "class_names": ["LowVeg", "ImpSurf", "Vehicle", "UrbanFurni", "Roof", "Facade", "Shrub", "Tree", "Soil/Gravel", "VertSurf", "Chimney"],
          "metrics": ["OA", "P", "R", "F1", "IoU", "wP", "wR", "wF1", "wIoU", "MCC", "Kappa"],
          "class_metrics": ["P", "R", "F1", "IoU"],
          "report_path": "*report/global_eval.log",
          "class_report_path": "*report/class_eval.log",
          "confusion_matrix_report_path" : "*report/confusion_matrix.log",
          "confusion_matrix_plot_path" : "*plot/confusion_matrix.svg",
          "class_distribution_report_path": "*report/class_distribution.log",
          "class_distribution_plot_path": "*plot/class_distribution.svg"
        },
        {
            "eval": "ClassificationUncertaintyEvaluator",
            "class_names": ["LowVeg", "ImpSurf", "Vehicle", "UrbanFurni", "Roof", "Facade", "Shrub", "Tree", "Soil/Gravel", "VertSurf", "Chimney"],
            "include_probabilities": true,
            "include_weighted_entropy": true,
            "include_clusters": true,
            "weight_by_predictions": false,
            "num_clusters": 10,
            "clustering_max_iters": 128,
            "clustering_batch_size": 1000000,
            "clustering_entropy_weights": false,
            "clustering_reduce_function": "mean",
            "gaussian_kernel_points": 256,
            "report_path": "*uncertainty/uncertainty.las",
            "plot_path": "*uncertainty/"
        }
      ]
    }





Predictions after the second training process JSON
----------------------------------------------------

The JSON below can be used to classify the validation point cloud with the
model after the second training process. It also computes the evaluation of the
model and the uncertainty assessment.

.. code-block:: json

    {
      "in_pcloud": [
        "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/mined/Mar18_val_hsv_std.laz"
      ],
      "out_pcloud": [
        "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/out/kpc_ATRY7/T2/preds/*"
      ],
      "sequential_pipeline": [
        {
          "predict": "PredictivePipeline",
          "model_path": "/hei/Hessigheim_Benchmark/Epoch_March2018/vl3d/out/kpc_ATRY7/T2/pipe/KPC_T2.pipe"
        },
        {
            "writer": "ClassifiedPcloudWriter",
            "out_pcloud": "*predicted.las"
        },
        {
          "writer": "PredictionsWriter",
          "out_preds": "*predictions.lbl"
        },
        {
          "eval": "ClassificationEvaluator",
          "class_names": ["LowVeg", "ImpSurf", "Vehicle", "UrbanFurni", "Roof", "Facade", "Shrub", "Tree", "Soil/Gravel", "VertSurf", "Chimney"],
          "metrics": ["OA", "P", "R", "F1", "IoU", "wP", "wR", "wF1", "wIoU", "MCC", "Kappa"],
          "class_metrics": ["P", "R", "F1", "IoU"],
          "report_path": "*report/global_eval.log",
          "class_report_path": "*report/class_eval.log",
          "confusion_matrix_report_path" : "*report/confusion_matrix.log",
          "confusion_matrix_plot_path" : "*plot/confusion_matrix.svg",
          "class_distribution_report_path": "*report/class_distribution.log",
          "class_distribution_plot_path": "*plot/class_distribution.svg"
        },
        {
            "eval": "ClassificationUncertaintyEvaluator",
            "class_names": ["LowVeg", "ImpSurf", "Vehicle", "UrbanFurni", "Roof", "Facade", "Shrub", "Tree", "Soil/Gravel", "VertSurf", "Chimney"],
            "include_probabilities": true,
            "include_weighted_entropy": true,
            "include_clusters": true,
            "weight_by_predictions": false,
            "num_clusters": 10,
            "clustering_max_iters": 128,
            "clustering_batch_size": 1000000,
            "clustering_entropy_weights": false,
            "clustering_reduce_function": "mean",
            "gaussian_kernel_points": 256,
            "report_path": "*uncertainty/uncertainty.las",
            "plot_path": "*uncertainty/"
        }
      ]
    }




Quantification
=================

Global results
-----------------
The table below shows the global evaluation metrics for the model after the
first training process on the validation point cloud.

.. csv-table::
    :file: ../../csv/kpconv_hessig_global_eval_T1.csv
    :widths: 9 9 9 9 9 9 9 9 9 9 9
    :header-rows: 1


The table below shows the global evaluation metrics for the model after the
second training process on the validation point cloud.

.. csv-table::
    :file: ../../csv/kpconv_hessig_global_eval_T2.csv
    :widths: 9 9 9 9 9 9 9 9 9 9 9
    :header-rows: 1


Class-wise results
---------------------
The table below shows the class-wise evaluation metrics for the model after
the first training process on the validation point cloud.

.. csv-table::
    :file: ../../csv/kpconv_hessig_class_eval_T1.csv
    :widths: 30 17 17 17 17
    :header-rows: 1


The table below shows the class-wise evaluation metrics for the model after
the second training process on the validation point cloud.

.. csv-table::
    :file: ../../csv/kpconv_hessig_class_eval_T2.csv
    :widths: 30 17 17 17 17
    :header-rows: 1





Visualization
================

The figure below shows the results of the model after the second training
process on the validation point cloud.

.. figure:: ../../img/kpconv_hessig_T2.png
    :scale: 40
    :alt: Figure representing the references, predictions, entropies, and
        likelihoods.

    Visualization of the reference and predictions, together with the binary
    fail/success mask. Also, the point-wise entropy and the likelihood for the
    roof, vehicle, facade, tree, and chimney classes.




Application
================

This example has two main applications:

#.  **Baseline model** for urban semantic segmentation in 3D point clouds with
    deep learning.

#.  Exploring **continue training** for deep learning models. In other words,
    the example illustrates how to train a model in more than one training
    process. This can be especially helpful when there are many big labeled
    point clouds so the model can learn how to classify considering different
    representations of the target classes, potentially improving its
    generalization.