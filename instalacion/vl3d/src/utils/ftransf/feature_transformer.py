# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
import numpy as np


# ---   EXCEPTIONS   --- #
# ---------------------- #
class FeatureTransformerException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to feature transformation components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class FeatureTransformer:
    """
    :author: Alberto M. Esmoris Pena

    Class for feature transformation operations.

    :ivar fnames: The names of the features to be transformed (by default).
    :vartype fnames: list or tuple
    :ivar report_path: The path to write the report file reporting the behavior
        of the transformer.
    :vartype report_path: str
    :ivar plot_path: The path to write the plot file. From some feature
        transformers it might be the path to the directory where many plots
        will be stored.
    :vartype plot_path: str
    :ivar selected_features: Either boolean mask or list of indices
        corresponding to the selected features (columns of the feature matrix).
    :vartype selected_features: list
    :ivar update_and_preserve: Boolean flag to control whether to discard
        all features but the transformed/updated ones (False, default) or
        to update the transformed features and preserve all the other
        features (True).
    :vartype update_and_preserve: bool
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ftransf_args(spec):
        """
        Extract the arguments to initialize/instantiate a FeatureTransformer
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a FeatureTransformer.
        """
        # Initialize
        kwargs = {
            'fnames': spec.get('fnames', None),
            'report_path': spec.get('report_path', None),
            'plot_path': spec.get('plot_path', None),
            'update_and_preserve': spec.get('update_and_preserve', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a FeatureTransformer.

        :param kwargs: The attributes for the FeatureTransformer.
        """
        # Fundamental initialization of any feature transformer
        self.fnames = kwargs.get('fnames', None)
        self.report_path = kwargs.get('report_path', None)
        self.plot_path = kwargs.get('plot_path', None)
        self.update_and_preserve = kwargs.get('update_and_preserve', False)
        self.selected_features = None

    # ---  FEATURE TRANSFORM METHODS  --- #
    # ----------------------------------- #
    @abstractmethod
    def transform(self, F, y=None, fnames=None, out_prefix=None):
        """
        The fundamental transformation logic defining the feature transformer.

        :param F: The input matrix of features to be imputed.
        :type F: :class:`np.ndarray`
        :param y: The vector of point-wise classes.
        :type y: :class:`np.ndarray`
        :param fnames: The list of features to be transformed. If None, it will
            be taken from the internal fnames of the feature transformer. If
            those are None too, then an exception will raise.
        :type fnames: list or tuple
        :param out_prefix: The output prefix (OPTIONAL). It might be used by a
            report to particularize the output path.
        :type out_prefix: str
        :return: The transformed matrix of features.
        :rtype: :class:`np.ndarray`
        """
        pass

    def transform_pcloud(self, pcloud, fnames=None, out_prefix=None):
        """
        Apply the transform method to a point cloud.

        See :meth:`feature_transformer.FeatureTransformer.transform`

        :param pcloud: The point cloud to be transformed.
        :type pcloud: :class:`.PointCloud`
        :param fnames: The list of features to be transformed. If None, it will
            be taken from the internal fnames of the feature transformer. If
            those are None too, then an exception will raise.
        :type fnames: list or tuple
        :param out_prefix: The output prefix (OPTIONAL). It might be used by a
            report to particularize the output path.
        :type out_prefix: str
        :return: A new point cloud that is the transformed version of the
            input point cloud.
        :rtype: :class:`.PointCloud`
        """
        # Check feature names
        fnames = self.safely_handle_fnames(fnames=fnames)
        # Transform features
        F = self.transform(
            pcloud.get_features_matrix(fnames),
            y=pcloud.get_classes_vector(),
            fnames=fnames,
            out_prefix=out_prefix
        )
        fnames = self.get_names_of_transformed_features(fnames=fnames)
        # Return output
        if self.update_and_preserve:
            # Return updated point cloud
            return pcloud\
                .remove_features(self.fnames)\
                .add_features(fnames, F, ftypes=F.dtype)
        else:
            # Return new point cloud
            return PointCloudFactoryFacade.make_from_arrays(
                pcloud.get_coordinates_matrix(),
                F,
                y=pcloud.get_classes_vector(),
                header=self.build_new_las_header(pcloud),
                fnames=fnames
            )

    def report(self, report, out_prefix=None):
        """
        Handle the way a report is reported. First, it will be reported using
        the logging system. Then, it will be written to a file if the
        transformer has a not None report_path.

        :param report: The report to be reported.
        :param out_prefix: The output prefix in case the output path must be
            expanded.
        :return: Nothing.
        """
        LOGGING.LOGGER.info(report)
        if self.report_path is not None:
            report.to_file(self.report_path, out_prefix=out_prefix)

    def build_new_las_header(self, pcloud):
        """
        Build the LAS header for the output point cloud.

        See :class:`.PointCloud` and
        :meth:`feature_transformer.FeatureTransformer.transform_pcloud`.

        :param pcloud: The input point cloud as reference to build the header
            for the new point cloud.
        :type pcloud: :class:`.PointCloud`
        :return: The LAS header for the output point cloud.
        """
        return pcloud.las.header

    def get_names_of_transformed_features(self, **kwargs):
        """
        Obtain the names that correspond to the transformed features.

        :return: The list of strings representing the names of the transformed
            features.
        :rtype: list
        """
        # Get feature names
        fnames = kwargs.get('fnames', None)
        # Validate feature names
        if fnames is None:
            raise FeatureTransformerException(
                'By default, a feature transformer requires base feature '
                'names to get the names of the transformed features.'
            )
        # Build names of transformed features from feature names
        return np.array(fnames)[self.selected_features].tolist()

    def safely_handle_fnames(self, fnames=None):
        """
        Handle given fnames in a safe-way, i.e., raising an exception if no
        valid fnames cannot be figured out from given input and the current
        state of the internal variables.

        :param fnames: Given input feature names.
        :type fnames: list of str
        :return: The handled feature names.
        :rtype: list of str
        """
        if fnames is None:
            if self.fnames is None:
                raise FeatureTransformerException(
                    'The features of a point cloud cannot be transformed if '
                    'they are not specified.'
                )
            else:
                fnames = self.fnames
        return fnames
