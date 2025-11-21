# The following was derived from an example module template from a PerkLabBootcamp slicer scripted module example file:
# https://github.com/PerkLab/PerkLabBootcamp/blob/master/Examples/CampTutorial2/CampTutorial2.py

# This template was used as a base to be expanded on, and will potentially be completely replaced
# as we work on the project code

"""
Guide for importing module to 3DSlicer:
3DSlicer > Edit > Application Settings > Modules > Additional Module Paths
> Add C:path/to/repo/CISC472-DBS-Target-Locator-Module-Fall-2025/DBSTargetLocator as path

Module can then be searched as "DBS Target Locator" or found in module dropdown under neuroscience heading
"""

import os
import logging
import numpy as np
import vtk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


#
# DBS Target Locator Module
#

class DBSTargetLocator(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    # Automated localization of candidate DBS sites for depression using fMRI.

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "DBS Target Locator"
        self.parent.categories = ["Neuroscience", "fMRI Analysis"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Gabriel Burrows, Leopold Ehrlich, Keiichiro Hayashi, Robert He, Amanda Zhu (Queen's University)"
        ]
        self.parent.helpText = """
        Automated localization of candidate electrode sites for deep brain stimulation in depression.
        This tool uses patient-specific fMRI data to highlight regions with abnormal activity that could
        inform DBS planning.
        """
        self.parent.acknowledgementText = ("Developed for group project in CISC 472 at Queen's University. "
                                           "Contains sample data from The Transdiagnostic Connectome Project at"
                                           "https://openneuro.org/datasets/ds005237")


        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in DBS Target Locator module
#

def registerSampleData():
    """
    Register OpenNeuro MRI and fMRI datasets (with JSON metadata)
    for subject NDARINVAG023WG3 from ds005237.
    """
    import SampleData
    import os

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # Anatomical MRI (T1w)
    # Load the anatomical T1-weighted MRI scan with its metadata
    anat_nii = "https://s3.amazonaws.com/openneuro.org/ds005237/sub-NDARINVAG023WG3/anat/sub-NDARINVAG023WG3_run-01_T1w.nii.gz"
    anat_json = "https://s3.amazonaws.com/openneuro.org/ds005237/sub-NDARINVAG023WG3/anat/sub-NDARINVAG023WG3_run-01_T1w.json"

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="OpenNeuro ds005237",
        sampleName="sub-NDARINVAG023WG3_T1w (Anatomical MRI)",
        thumbnailFileName=os.path.join(iconsPath, "OpenNeuroAnat.png"),
        uris=[anat_nii, anat_json],
        fileNames=[
            "sub-NDARINVAG023WG3_run-01_T1w.nii.gz",
            "sub-NDARINVAG023WG3_run-01_T1w.json"
        ],
        nodeNames=["sub-NDARINVAG023WG3_T1w", None],  # only load the NIfTI file
        checksums=None
    )

    # Functional MRI (BOLD)
    # Load the functional BOLD fMRI scan with its metadata
    func_nii = "https://s3.amazonaws.com/openneuro.org/ds005237/sub-NDARINVAG023WG3/func/sub-NDARINVAG023WG3_task-restAP_run-01_bold.nii.gz"
    func_json = "https://s3.amazonaws.com/openneuro.org/ds005237/sub-NDARINVAG023WG3/func/sub-NDARINVAG023WG3_task-restAP_run-01_bold.json"

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="OpenNeuro ds005237",
        sampleName="sub-NDARINVAG023WG3_task-restAP (fMRI)",
        thumbnailFileName=os.path.join(iconsPath, "OpenNeurofMRI.png"),
        uris=[func_nii, func_json],
        fileNames=[
            "sub-NDARINVAG023WG3_task-restAP_run-01_bold.nii.gz",
            "sub-NDARINVAG023WG3_task-restAP_run-01_bold.json"
        ],
        nodeNames=["sub-NDARINVAG023WG3_task-restAP_bold", None],
        checksums=None
    )

#
# DBSTargetLocatorWidget
#

class DBSTargetLocatorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """
    The user interface for the DBS Target Locator module.
    This class handles all the UI elements and user interactions.
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        Sets up the basic properties we'll need throughout the widget's lifecycle.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        This is where we build the entire user interface and connect all the buttons and controls.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/DBSTargetLocator.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. This allows the selectors to show available nodes.
        self.ui.inputAnatomicalSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.inputFunctionalSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.outputVolumeSelector.setMRMLScene(slicer.mrmlScene)

        # Create logic class. This handles all the actual computation work.
        self.logic = DBSTargetLocatorLogic()

        # Observers - these let us respond to scene events like opening/closing scenes
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Connect UI elements - wire up all the buttons and controls to their callback functions
        self.ui.inputAnatomicalSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputAnatomicalSelected)
        self.ui.inputFunctionalSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputFunctionalSelected)
        self.ui.outputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputVolumeSelected)
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()


    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        Clean up any observers we've created to prevent memory leaks.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        Make sure the parameter node exists and is up to date.
        """
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        Remove observers to prevent callbacks when we're not active.
        """
        if self._parameterNode:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller=None, event=None):
        """
        Called when the MRML scene starts closing.
        We can do cleanup tasks here if needed.
        """
        self.setParameterNode(None)

    def onSceneEndClose(self, caller=None, event=None):
        """
        Called after the MRML scene finishes closing.
        Reinitialize everything so the module is ready to use again.
        """
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure a parameter node exists and reference the first markup if available.
        The parameter node stores all the settings and references for this module.
        """
        self.setParameterNode(self.logic.getParameterNode())

        # If no anatomical volume is selected yet, try to use the first scalar volume in the scene
        if not self._parameterNode.GetNodeReference(self.logic.INPUT_ANATOMICAL):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID(self.logic.INPUT_ANATOMICAL, firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set the parameter node and update observers to keep the GUI in sync.
        This ensures that when the parameter node changes, the UI updates automatically.
        """
        if self._parameterNode:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Update the GUI to match the current parameter node
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        Synchronize GUI elements with the current parameter node values.
        This is called whenever the parameter node is modified, ensuring the UI always
        reflects the current state of the module.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

            # Prevent recursive updates - we don't want GUI changes triggering more GUI updates
        self._updatingGUIFromParameterNode = True

        # Update each UI element to match the parameter node
        anatomicalNode = self._parameterNode.GetNodeReference(self.logic.INPUT_ANATOMICAL)
        functionalNode = self._parameterNode.GetNodeReference(self.logic.INPUT_FUNCTIONAL)
        outputNode = self._parameterNode.GetNodeReference(self.logic.OUTPUT_VOLUME)

        self.ui.inputAnatomicalSelector.setCurrentNode(anatomicalNode)
        self.ui.inputFunctionalSelector.setCurrentNode(functionalNode)
        self.ui.outputVolumeSelector.setCurrentNode(outputNode)

        # Enable the Apply button only if we have the required inputs
        self.ui.applyButton.enabled = anatomicalNode is not None and functionalNode is not None

        # All done updating, allow callbacks again
        self._updatingGUIFromParameterNode = False

    def onInputAnatomicalSelected(self, newNode):
        """
        Called when the user selects a different anatomical volume from the dropdown.
        Updates the parameter node to track the new selection.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        if newNode is None:
            self._parameterNode.SetNodeReferenceID(self.logic.INPUT_ANATOMICAL, None)
        else:
            self._parameterNode.SetNodeReferenceID(self.logic.INPUT_ANATOMICAL, newNode.GetID())

    def onInputFunctionalSelected(self, newNode):
        """
        Called when the user selects a different functional volume from the dropdown.
        Updates the parameter node to track the new selection.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        if newNode is None:
            self._parameterNode.SetNodeReferenceID(self.logic.INPUT_FUNCTIONAL, None)
        else:
            self._parameterNode.SetNodeReferenceID(self.logic.INPUT_FUNCTIONAL, newNode.GetID())

    def onOutputVolumeSelected(self, newNode):
        """
        Called when the user selects a different output volume from the dropdown.
        Updates the parameter node to track the new selection.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        if newNode is None:
            self._parameterNode.SetNodeReferenceID(self.logic.OUTPUT_VOLUME, None)
        else:
            self._parameterNode.SetNodeReferenceID(self.logic.OUTPUT_VOLUME, newNode.GetID())

    def onApplyButton(self):
        """
        Called when the user clicks the Apply button.
        Triggers the fMRI analysis using the selected anatomical and functional volumes.
        """
        try:
            # Get the input volumes
            anatomicalVolume = self.ui.inputAnatomicalSelector.currentNode()
            functionalVolume = self.ui.inputFunctionalSelector.currentNode()
            outputVolume = self.ui.outputVolumeSelector.currentNode()

            # Run the analysis
            self.logic.analyzeFMRI(anatomicalVolume, functionalVolume, outputVolume)

            # Show completion message
            slicer.util.infoDisplay("fMRI analysis completed successfully!")

        except Exception as e:
            slicer.util.errorDisplay("Failed to complete analysis: " + str(e))
            import traceback
            traceback.print_exc()

#
# DBSTargetLocatorLogic
#

class DBSTargetLocatorLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
    """
    This class implements all the actual computation for the module.
    The interface is designed so other Python code can import this class and use its
    functionality without needing the GUI widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    # Constants for parameter node keys - these help us store and retrieve settings
    INPUT_ANATOMICAL = "InputAnatomical"
    INPUT_FUNCTIONAL = "InputFunctional"
    OUTPUT_VOLUME = "OutputVolume"

    def __init__(self):
        """
        Called when the logic class is instantiated.
        Initialize member variables that we'll use throughout the logic's lifecycle.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        VTKObservationMixin.__init__(self)



    def analyzeFMRI(self, anatomicalVolume, functionalVolume, outputVolume=None):
        """
        Perform fMRI analysis to identify candidate DBS target locations.

        This is the main analysis function where we'll implement our DBS target localization algorithm.
        It takes anatomical (T1w) and functional (BOLD) MRI data as input and produces an output
        volume highlighting regions of interest for potential DBS electrode placement.

        Parameters:
        - anatomicalVolume: The T1-weighted anatomical MRI volume node
        - functionalVolume: The BOLD fMRI volume node (4D timeseries)
        - outputVolume: Optional output volume node for results. If None, a new one is created.

        Returns:
        - The output volume node containing the analysis results

        TODO: Implement our fMRI analysis pipeline here, may include:
        - Preprocessing (motion correction, spatial smoothing, temporal filtering)
        - Statistical analysis (activation maps, connectivity analysis)
        - Region of interest identification
        - Target localization based on clinical criteria
        """

        # Validate inputs
        if not anatomicalVolume or not functionalVolume:
            raise ValueError("Both anatomical and functional volumes are required")

        logging.info("Starting fMRI analysis...")
        logging.info(f"Anatomical volume: {anatomicalVolume.GetName()}")
        logging.info(f"Functional volume: {functionalVolume.GetName()}")

        # TODO: Implement registration of anatomical mri with fmri before analysis
        # look into use of BRAINSFit module for this
        # https://www.slicer.org/w/index.php/Documentation/Nightly/Modules/BRAINSFit

        # Create output volume if not provided
        if not outputVolume:
            outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "DBSTargetMap")
            logging.info(f"Created output volume: {outputVolume.GetName()}")

        # TODO: Add our fMRI analysis implementation here
        # This is where we'll implement the core functionality for:
        # 1. Loading and preprocessing the fMRI data
        # 2. Analyzing functional connectivity or activation patterns
        # 3. Identifying candidate target regions based on our criteria
        # 4. Creating a visualization/map of potential DBS targets

        # For now, we'll just copy the anatomical volume to demonstrate data flow
        # Replace this with actual analysis pipeline
        logging.info("Running analysis (placeholder implementation)...")

        # Get the anatomical volume data
        anatomicalArray = slicer.util.arrayFromVolume(anatomicalVolume)

        # TODO: Process the functional volume
        # functionalArray = slicer.util.arrayFromVolume(functionalVolume)
        # Note: Functional volume is 4D (time series), so we'll need to handle the temporal dimension

        # TODO: Implement our analysis algorithm here
        # For now, just creates a placeholder output based on the anatomical volume
        outputArray = np.copy(anatomicalArray)

        # Update the output volume with the results
        slicer.util.updateVolumeFromArray(outputVolume, outputArray)
        outputVolume.CopyOrientation(anatomicalVolume)

        logging.info("fMRI analysis completed")

        return outputVolume

    def preprocessFunctionalData(self, functionalVolume):
        """
        Preprocess the functional MRI data before analysis.

        TODO: Implement preprocessing steps such as:
        - Motion correction
        - Slice timing correction
        - Spatial smoothing
        - Temporal filtering (high-pass, low-pass)
        - Normalization
        This can be done with use of the information provided in the JSON sidecar files
        associated with each mri scan in our data set

        Parameters:
        - functionalVolume: The raw BOLD fMRI volume node

        Returns:
        - Preprocessed functional volume
        """
        logging.info("Preprocessing functional data (not yet implemented)...")
        # TODO: Add preprocessing implementation
        pass

    def identifyTargetRegions(self, analysisVolume, threshold=None):
        """
        Identify and rank potential DBS target regions based on analysis results.

        TODO: Implement target identification logic based on:
        - Statistical thresholds
        - Anatomical constraints
        - Clinical criteria for DBS in depression
        We will likely use healthy scans to develop a baseline and compare scans from patients with
        depression against the baseline.

        Parameters:
        - analysisVolume: The volume containing analysis results
        - threshold: Optional threshold for region selection

        Returns:
        - List of candidate target regions with coordinates and metrics
        """
        logging.info("Identifying target regions (not yet implemented)...")
        # TODO: Add target identification implementation
        pass

    def loadJSONMetadata(self, jsonFilePath):
        """
        Load and parse JSON sidecar metadata from BIDS-formatted neuroimaging data.

        The JSON files contain important parameters like:
        - RepetitionTime (TR)
        - EchoTime (TE)
        - FlipAngle
        - SliceTiming
        that we will need to consider when working with this data

        Parameters:
        - jsonFilePath: Path to the JSON metadata file

        Returns:
        - Dictionary containing the parsed metadata
        """
        import json

        if not os.path.exists(jsonFilePath):
            logging.warning(f"JSON metadata file not found: {jsonFilePath}")
            return {}

        try:
            with open(jsonFilePath, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Loaded metadata from: {jsonFilePath}")
            return metadata
        except Exception as e:
            logging.error(f"Failed to load JSON metadata: {str(e)}")
            return {}

#
# DBSTargetLocatorTest
#

class DBSTargetLocatorTest(ScriptedLoadableModuleTest):
    """
    This is the test case for our scripted module.
    Tests verify that the module loads data correctly and performs basic operations.
    """

    def setUp(self):
        """
        Reset the state before each test.
        Clearing the scene ensures tests are independent and repeatable.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """
        Run the test suite.
        We can add multiple test methods here as our module grows.
        """
        self.setUp()
        self.test_DBSTargetLocator_DataLoading()
        self.test_DBSTargetLocator_BasicAnalysis()

    def test_DBSTargetLocator_DataLoading(self):
        """
        Test that the OpenNeuro sample data can be loaded correctly.
        This verifies the data source registration and download mechanism.
        """
        self.delayDisplay("Starting data loading test")

        # Register the sample data sources
        registerSampleData()

        # Try to download the anatomical MRI
        import SampleData
        try:
            anatomicalVolume = SampleData.downloadSample('sub-NDARINVAG023WG3_T1w (Anatomical MRI)')
            self.assertIsNotNone(anatomicalVolume, "Failed to load anatomical volume")
            self.delayDisplay("Successfully loaded anatomical MRI")
        except Exception as e:
            logging.warning(f"Could not download anatomical data: {str(e)}")
            logging.warning("This may be expected if you don't have internet connectivity")

        # Try to download the functional MRI
        try:
            functionalVolume = SampleData.downloadSample('sub-NDARINVAG023WG3_task-restAP (fMRI)')
            self.assertIsNotNone(functionalVolume, "Failed to load functional volume")
            self.delayDisplay("Successfully loaded functional MRI")
        except Exception as e:
            logging.warning(f"Could not download functional data: {str(e)}")
            logging.warning("This may be expected if you don't have internet connectivity")

        self.delayDisplay('Data loading test completed')

    def test_DBSTargetLocator_BasicAnalysis(self):
        """
        Test that the basic analysis pipeline runs without errors.
        This uses synthetic test data to verify the logic flow.
        """
        self.delayDisplay("Starting basic analysis test")

        # Create synthetic test volumes
        imageSize = [64, 64, 64]
        imageSpacing = [1.0, 1.0, 1.0]
        imageOrigin = [0.0, 0.0, 0.0]

        # Create anatomical volume
        anatImageArray = np.full(imageSize, 100, dtype=np.uint8)
        # Create the node from the array
        anatomicalVolume = slicer.util.addVolumeFromArray(anatImageArray)
        # Set properties on the node object
        anatomicalVolume.SetName("TestAnatomical")
        anatomicalVolume.SetSpacing(imageSpacing)
        anatomicalVolume.SetOrigin(imageOrigin)

        # Create "functional" volume (using 3D for simplicity but real data is 4D fmri)
        funcImageArray = np.full(imageSize, 50, dtype=np.uint8)
        # Create the node from the array
        functionalVolume = slicer.util.addVolumeFromArray(funcImageArray)
        # Set properties on the node object
        functionalVolume.SetName("TestFunctional")
        functionalVolume.SetSpacing(imageSpacing)
        functionalVolume.SetOrigin(imageOrigin)

        # Check that volumes were created
        self.assertIsNotNone(anatomicalVolume)
        self.assertIsNotNone(functionalVolume)

        # Create and test the logic
        logic = DBSTargetLocatorLogic()

        # Run the analysis
        try:
            outputVolume = logic.analyzeFMRI(anatomicalVolume, functionalVolume)
            self.assertIsNotNone(outputVolume, "Analysis did not produce output volume")

            # Verify the output (in this base version, it's a copy of anatomical)
            # we will need to generate more robust test cases once the DBS target locator logic is implemented
            # that include true 4D fmri data
            outputArray = slicer.util.arrayFromVolume(outputVolume)
            self.assertTrue(np.array_equal(outputArray, anatImageArray))

            self.delayDisplay("Analysis completed successfully")
        except Exception as e:
            self.fail(f"Analysis failed with error: {str(e)}")

        self.delayDisplay('Basic analysis test passed!')