# The following is an example module template taken from PerkLabBootcamp Example file:
# https://github.com/PerkLab/PerkLabBootcamp/blob/master/Examples/CampTutorial2/CampTutorial2.py

# It is to be used as a base to expand on, and potentially completely replace, as we work on the project code

"""
Guide for importing module to 3DSlicer:
3DSlicer > Edit > Application Settings > Modules > Additional Module Paths
> Add C:path/to/repo/CISC472-DBS-Target-Locator-Module-Fall-2025/DBSTargetLocator as path

Module can then be searched as "DBS Target Locator" or found in module dropdown under neuroscience heading
"""

import os
import logging
import vtk, qt, ctk, slicer
import numpy as np
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

from slicer import qMRMLWidget
import qt

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
        self.parent.acknowledgementText = "Developed for group project in CISC 472 at Queen's University."

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in DBS Target Locator module
#


def registerSampleData():
    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="DBSTargetLocator",
        sampleName="DBSTargetLocator1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "DBSTargetLocator1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="DBSTargetLocator1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="DBSTargetLocator1"
    )

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="DBSTargetLocator",
        sampleName="DBSTargetLocator2",
        thumbnailFileName=os.path.join(iconsPath, "DBSTargetLocator2.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="DBSTargetLocator2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        nodeNames="DBSTargetLocator2"
    )

#
# DBSTargetLocatorWidget
#

class DBSTargetLocatorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self) # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/DBSTargetLocator.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        self.ui.inputMarkupSelector.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = DBSTargetLocatorLogic()
        self.logic.setupScene()

        # Observers
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Connect UI elements
        self.ui.inputMarkupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputMarkupSelected)
        self.ui.opacitySliderWidget.connect("valueChanged(double)", self.onOpacitySliderChanged)
        self.ui.autoUpdateCheckBox.connect("clicked(bool)", self.onAutoUpdateClicked)
        self.ui.outputLineEdit.connect("currentPathChanged(QString)", self.onOutputPathChanged)
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.exportDataButton.connect('clicked(bool)', self.onExportButtonClicked)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        pathValue = self.logic.getExportPath()
        if pathValue:
            self.ui.outputLineEdit.setCurrentPath(pathValue)

    def onExportButtonClicked(self):
        """Export the current sphere model to disk."""
        self.logic.exportSphereModel()

    def onOutputPathChanged(self, newPath):
        """Update the file path used for exporting the sphere model."""
        self.logic.setExportPath(newPath)

    def onOpacitySliderChanged(self, newValue):
        """Update the sphere model opacity and optionally refresh automatically."""
        if slicer.mrmlScene.IsImporting():
            return
        self.logic.setOpacity(newValue)
        if self.ui.autoUpdateCheckBox.checked:
            self.onApplyButton()

    def onAutoUpdateClicked(self, checked):
        """Toggle automatic updates of the sphere model when markup points change."""
        if slicer.mrmlScene.IsImporting():
            return
        self.logic.setAutoUpdate(checked)
        self.onApplyButton()

    def onApplyButton(self):
        """Manually trigger the sphere model update using current inputs."""
        try:
            self.logic.updateSphere(self.ui.inputMarkupSelector.currentNode(), self.ui.opacitySliderWidget.value)
        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: " + str(e))
            import traceback
            traceback.print_exc()

    def initializeParameterNode(self):
        """Ensure a parameter node exists and reference the first markup if available."""
        self.setParameterNode(self.logic.getParameterNode())
        if not self._parameterNode.GetNodeReference(self.logic.INPUT_MARKUP):
            firstMarkupNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstMarkupNode:
                self._parameterNode.SetNodeReferenceID(self.logic.INPUT_MARKUP, firstMarkupNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """Set the parameter node and update observers to keep the GUI in sync."""
        if self._parameterNode:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """Synchronize GUI elements with the current parameter node values."""
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        self._updatingGUIFromParameterNode = True
        inputNode = self._parameterNode.GetNodeReference(self.logic.INPUT_MARKUP)
        self.ui.inputMarkupSelector.setCurrentNode(inputNode)
        self.ui.opacitySliderWidget.value = self.logic.getOpacity()
        self.ui.autoUpdateCheckBox.setChecked(self.logic.getAutoUpdate())
        self._updatingGUIFromParameterNode = False

    def onInputMarkupSelected(self, newNode):
        """Update the parameter node when the user selects a different markup node."""
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        if newNode is None:
            self._parameterNode.SetNodeReferenceID(self.logic.INPUT_MARKUP, None)
            self.logic.setAutoUpdate(False)
        else:
            self._parameterNode.SetNodeReferenceID(self.logic.INPUT_MARKUP, newNode.GetID())

    def onSceneStartClose(self, caller=None, event=None):
        """Handle tasks when the MRML scene starts closing."""
        # e.g., clear references to nodes or UI
        pass

    def onSceneEndClose(self, caller=None, event=None):
        """Reinitialize the parameter node after the MRML scene finishes closing."""
        # e.g., re-initialize parameter node
        self.initializeParameterNode()

#
# DBSTargetLocatorLogic
#

class DBSTargetLocatorLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    OUTPUT_PATH_SETTING = "DBSTargetLocator/OutputPath"
    INPUT_MARKUP = "InputMarkup"
    SPHERE_MODEL = "SphereModel"
    OPACITY = "Opacity"
    OPACITY_DEFAULT = 0.8
    AUTOUPDATE = "AutoUpdate"
    AUTOUPDATE_DEFAULT = False

    def __init__(self):
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        VTKObservationMixin.__init__(self)
        self.fiducialNode = None
        self.sphereNode = None
        self.observedMarkupNode = None
        self.isImporting = False

    def setupScene(self):
        """Create the sphere model if needed and set up scene observers."""
        parameterNode = self.getParameterNode()
        sphereModel = parameterNode.GetNodeReference(self.SPHERE_MODEL)
        if not sphereModel:
            sphereModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.SPHERE_MODEL)
            sphereModel.CreateDefaultDisplayNodes()
            parameterNode.SetNodeReferenceID(self.SPHERE_MODEL, sphereModel.GetID())
        self.addObserver(slicer.mrmlScene, slicer.vtkMRMLScene.StartImportEvent, self.onSceneImportStart)
        self.addObserver(slicer.mrmlScene, slicer.vtkMRMLScene.EndImportEvent, self.onSceneImportEnd)
        self.setAutoUpdate(self.getAutoUpdate())

    def exportSphereModel(self):
        """Save the sphere model to the file system at the configured export path."""
        parameterNode = self.getParameterNode()
        sphereNode = parameterNode.GetNodeReference(self.SPHERE_MODEL)
        if not sphereNode:
            logging.info("Cannot export sphere model, not created yet")
            return
        exportPath = slicer.util.settingsValue(self.OUTPUT_PATH_SETTING, "")
        fileName = sphereNode.GetName() + ".stl"
        fileFullName = os.path.join(exportPath, fileName)
        logging.info("Exporting sphere model to: {}".format(fileFullName))
        slicer.util.saveNode(sphereNode, fileFullName)

    def onSceneImportStart(self, caller, event):
        """Record nodes and set importing flag at the start of scene import."""
        self.isImporting = True
        parameterNode = self.getParameterNode()
        self.sphereNode = parameterNode.GetNodeReference(self.SPHERE_MODEL)
        self.fiducialNode = parameterNode.GetNodeReference(self.INPUT_MARKUP)

    def onSceneImportEnd(self, caller, event):
        """Restore sphere and markup nodes after scene import finishes."""
        parameterNode = self.getParameterNode()
        currentSphereNode = parameterNode.GetNodeReference(self.SPHERE_MODEL)
        if self.sphereNode != currentSphereNode:
            parameterNode.SetNodeReferenceID(self.SPHERE_MODEL, self.sphereNode.GetID())
            self.removeNode(currentSphereNode)

        currentMarkup = parameterNode.GetNodeReference(self.INPUT_MARKUP)
        if self.fiducialNode != currentMarkup:
            self.removeNode(self.fiducialNode)
            self.fiducialNode = currentMarkup

        self.isImporting = False
        self.setAutoUpdate(self.getAutoUpdate())
        self.updateSphere(currentMarkup, self.getOpacity())
        parameterNode.Modified()

    def removeNode(self, node):
        """Safely remove a node along with its display and storage nodes."""
        if node is None:
            return
        for i in range(node.GetNumberOfDisplayNodes()):
            slicer.mrmlScene.RemoveNode(node.GetNthDisplayNode(i))
        for i in range(node.GetNumberOfStorageNodes()):
            slicer.mrmlScene.RemoveNode(node.GetNthStorageNode(i))
        slicer.mrmlScene.RemoveNode(node)

    def setOpacity(self, newValue):
        """Store the sphere model opacity in the parameter node."""
        self.getParameterNode().SetParameter(self.OPACITY, str(newValue))

    def getOpacity(self):
        """Retrieve the current sphere model opacity, or default if unset."""
        value = self.getParameterNode().GetParameter(self.OPACITY)
        return float(value) if value else self.OPACITY_DEFAULT

    def setAutoUpdate(self, autoUpdate):
        """Enable or disable automatic updating of the sphere model."""
        parameterNode = self.getParameterNode()
        parameterNode.SetParameter(self.AUTOUPDATE, "true" if autoUpdate else "false")
        markupNode = parameterNode.GetNodeReference(self.INPUT_MARKUP)
        if self.observedMarkupNode:
            self.removeObserver(self.observedMarkupNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onMarkupsUpdated)
            self.observedMarkupNode = None
        if autoUpdate and markupNode:
            self.observedMarkupNode = markupNode
            self.addObserver(self.observedMarkupNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onMarkupsUpdated)

    def getAutoUpdate(self):
        """Return whether auto-update is enabled."""
        value = self.getParameterNode().GetParameter(self.AUTOUPDATE)
        if not value:
            return self.AUTOUPDATE_DEFAULT
        return value.lower() != "false"

    def setExportPath(self, newPath):
        """Store the export path for the sphere model in settings."""
        qt.QSettings().setValue(self.OUTPUT_PATH_SETTING, newPath)

    def getExportPath(self):
        """Retrieve the configured export path for the sphere model."""
        return slicer.util.settingsValue(self.OUTPUT_PATH_SETTING, None)

    def onMarkupsUpdated(self, caller, event):
        """Callback triggered when markup points are modified to update the sphere."""
        markupNode = self.getParameterNode().GetNodeReference(self.INPUT_MARKUP)
        self.updateSphere(markupNode, self.getOpacity())

    def updateSphere(self, inputMarkup, opacity):
        """Compute and update the sphere model based on the first two markup points."""
        parameterNode = self.getParameterNode()
        outputModel = parameterNode.GetNodeReference(self.SPHERE_MODEL)
        if not inputMarkup or not outputModel:
            return
        if inputMarkup.GetNumberOfControlPoints() < 2:
            return
        p0, p1 = np.zeros(3), np.zeros(3)
        inputMarkup.GetNthControlPointPosition(0, p0)
        inputMarkup.GetNthControlPointPosition(1, p1)
        center = (p0 + p1) / 2.0
        radius = np.linalg.norm(p1 - p0) / 2.0
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center)
        sphereSource.SetRadius(radius)
        sphereSource.Update()
        if outputModel.GetNumberOfDisplayNodes() < 1:
            outputModel.CreateDefaultDisplayNodes()
        outputModel.SetAndObservePolyData(sphereSource.GetOutput())
        outputModel.GetDisplayNode().SetOpacity(opacity)

#
# DBSTargetLocatorTest
#

class DBSTargetLocatorTest(ScriptedLoadableModuleTest):
    """Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_DBSTargetLocator1()

    def test_DBSTargetLocator1(self):
        """Test that the DBS Target Locator logic correctly creates and updates a sphere model from markup points."""
        self.delayDisplay("Starting test")
        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('DBSTargetLocator1')
        self.delayDisplay("Loaded sample data")

        logic = DBSTargetLocatorLogic()
        logic.setupScene()
        self.assertIsNotNone(inputVolume)

        # Minimal test: just make sure updateSphere runs
        markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        markupNode.AddControlPoint([0, 0, 0])
        markupNode.AddControlPoint([1, 1, 1])
        logic.updateSphere(markupNode, 0.5)
        outputModel = logic.getParameterNode().GetNodeReference(logic.SPHERE_MODEL)
        self.assertIsNotNone(outputModel)
        self.delayDisplay("Test passed")