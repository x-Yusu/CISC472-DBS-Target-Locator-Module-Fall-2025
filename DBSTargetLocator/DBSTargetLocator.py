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
import qt
import ctk
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
        This tool uses patient-specific fMRI connectivity matrices (.h5) to highlight regions with 
        abnormal activity relative to a healthy control baseline.
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


    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="OpenNeuro ds005237",
        sampleName="sub-NDARINVAG023WG3_T1w (Anatomical MRI)",
        thumbnailFileName=os.path.join(iconsPath, "OpenNeuroAnat.png"),
        uris=[anat_nii],
        fileNames=["sub-NDARINVAG023WG3_run-01_T1w.nii.gz"],
        nodeNames=["sub-NDARINVAG023WG3_T1w"],
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


        #
        # Input Data Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Patient Data Selection"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        #
        # Patient File Selector (H5)
        #
        self.patientSelector = ctk.ctkPathLineEdit()
        self.patientSelector.filters = ctk.ctkPathLineEdit.Files
        self.patientSelector.nameFilters = ["HDF5 Files (*.h5)"]
        self.patientSelector.settingKey = 'DBSTargetLocator/PatientPath'
        self.patientSelector.toolTip = "Select the processed .h5 timeseries file for the patient."
        parametersFormLayout.addRow("Patient fMRI (.h5):", self.patientSelector)

        #
        # Patient Anatomical Selector (Volume)
        #
        self.anatomicalSelector = slicer.qMRMLNodeComboBox()
        self.anatomicalSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.anatomicalSelector.selectNodeUponCreation = True
        self.anatomicalSelector.addEnabled = False
        self.anatomicalSelector.removeEnabled = False
        self.anatomicalSelector.noneEnabled = False
        self.anatomicalSelector.showHidden = False
        self.anatomicalSelector.showChildNodeTypes = False
        self.anatomicalSelector.setMRMLScene(slicer.mrmlScene)
        self.anatomicalSelector.toolTip = "Select the Patient's T1 Anatomical Volume (Loaded in Scene)."
        parametersFormLayout.addRow("Patient Anatomy (T1):", self.anatomicalSelector)

        #
        # Dependency Management Area
        #
        self.installDepsButton = qt.QPushButton("Install Python Dependencies")
        self.installDepsButton.toolTip = "Click this if you get import errors (installs pandas, h5py)."
        self.installDepsButton.enabled = True
        parametersFormLayout.addRow(self.installDepsButton)

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Calculate DBS Targets")
        self.applyButton.toolTip = "Run the analysis."
        self.applyButton.enabled = False
        parametersFormLayout.addRow(self.applyButton)

        # Add vertical spacer
        self.layout.addStretch(1)

        # Create logic class. This handles all the actual computation work.
        self.logic = DBSTargetLocatorLogic()

        # Connect UI elements - wire up all the buttons and controls to their callback functions
        self.patientSelector.connect("currentPathChanged(const QString &)", self.onSelect)
        self.anatomicalSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.installDepsButton.connect('clicked(bool)', self.onInstallDeps)

        # Check inputs initially
        self.onSelect()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        Clean up any observers we've created to prevent memory leaks.
        """
        self.removeObservers()

    def onSelect(self):
        """
        Enable the Apply button only if all inputs are ready.
        """
        self.applyButton.enabled = (
                self.patientSelector.currentPath != "" and
                self.anatomicalSelector.currentNode() is not None
        )

    def onInstallDeps(self):
        """
        Installs required python packages into Slicer's environment.
        """
        try:
            slicer.util.pip_install("pandas h5py tables scipy")
            slicer.util.infoDisplay("Dependencies installed! You may need to restart Slicer.")
        except Exception as e:
            slicer.util.errorDisplay(f"Installation failed: {e}")

    def onApplyButton(self):
        """
        Called when the user clicks the Apply button.
        Triggers the fMRI analysis using the selected anatomical and functional inputs.
        """
        try:
            patient_path = self.patientSelector.currentPath
            anatomical_node = self.anatomicalSelector.currentNode()

            # Run the analysis
            self.logic.analyzeFMRI(patient_path, anatomical_node)

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

    def __init__(self):
        """
        Called when the logic class is instantiated.
        Initialize member variables that we'll use throughout the logic's lifecycle.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        VTKObservationMixin.__init__(self)

    def get_module_resource_path(self):
        """
        Helper to find the module's Resources directory.
        """
        moduleDir = os.path.dirname(__file__)
        return os.path.join(moduleDir, 'Resources')

    def registerPatientToMNI(self, fixedMNI, movingPatient):
        """
        1. Register Patient (Moving) to MNI (Fixed).
        2. Create a transform node.
        3. Return the transform node (PatientToMNI).
        """
        logging.info("Starting Registration (BRAINSFit)...")

        # Create transform node
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "PatientToMNI_Transform")

        # Setup BRAINSFit parameters
        # need to tune these parameters if registration is poor (e.g., increase samplingPercentage)
        parameters = {
            "fixedVolume": fixedMNI.GetID(),
            "movingVolume": movingPatient.GetID(),
            "linearTransform": transformNode.GetID(),  # Output transform
            "useRigid": True,
            "useAffine": True,
            "useBSpline": False,
            "samplingPercentage": 0.02,
            "initializeTransformMode": "useMomentsAlign"
        }

        # Run CLI module
        brainsFit = slicer.modules.brainsfit
        cliNode = slicer.cli.runSync(brainsFit, None, parameters)

        if cliNode.GetStatus() & cliNode.ErrorsMask:
            errorMsg = cliNode.GetErrorText()
            raise RuntimeError(f"BRAINSFit registration failed: {errorMsg}")

        logging.info("Registration complete. Transform created.")
        return transformNode

    def analyzeFMRI(self, patient_h5_path, anatomicalVolume):
        """
        Perform fMRI analysis to identify candidate DBS target locations.
        """

        logging.info("Starting fMRI analysis...")
        logging.info(f"Anatomical volume: {anatomicalVolume.GetName()}")
        logging.info(f"Patient Data: {patient_h5_path}")

        import pandas as pd
        import numpy as np
        import h5py

        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)

        try:
            # ---------------------------------------------------------------
            # 0. SETUP PATHS
            # ---------------------------------------------------------------
            resourceDir = self.get_module_resource_path()
            control_dir = os.path.join(resourceDir, 'HealthyControls')
            atlas_path = os.path.join(resourceDir, 'Atlas', 'Master_DBS_Atlas.nii.gz')
            mni_template_path = os.path.join(resourceDir, 'Atlas', 'mni_icbm152_t1_tal_nlin_asym_09c.nii')

            # Consider checking if files exist and downloading them if missing (except Controls).
            if not os.path.exists(control_dir): raise FileNotFoundError(f"Healthy Controls not found at {control_dir}.")
            if not os.path.exists(atlas_path): raise FileNotFoundError(f"Master Atlas not found at {atlas_path}.")
            if not os.path.exists(mni_template_path): raise FileNotFoundError(
                f"MNI Template not found at {mni_template_path}.")

            # ---------------------------------------------------------------
            # 1. LOAD ATLAS & MNI TEMPLATE
            # ---------------------------------------------------------------
            # Load Master Atlas
            # NOTE currently Atlas is corrupt ("huge red regions"), visual alignment will fail.
            # Verify atlas dimensions match MNI template dimensions here.
            atlas_node = slicer.util.loadLabelVolume(atlas_path)
            atlas_node.SetName("Master_Atlas_MNI")
            atlas_data = slicer.util.arrayFromVolume(atlas_node)

            # Load MNI Template
            mni_node = slicer.util.loadVolume(mni_template_path)
            mni_node.SetName("MNI_Template_Ref")

            # Hide them initially
            atlas_node.GetDisplayNode().SetVisibility(0)
            mni_node.GetDisplayNode().SetVisibility(0)

            # ---------------------------------------------------------------
            # 2. COMPUTE Z-SCORES (On MNI Atlas Grid)
            # ---------------------------------------------------------------
            # Load Patient Metric (With Auto-merging AP/PA)
            logging.info(f"Loading Patient fMRI: {patient_h5_path}")
            patient_metric = self._load_and_compute_metric(patient_h5_path, pd, h5py, np)

            # Load/Calc Baseline
            baseline_file = os.path.join(control_dir, 'baseline_stats.npy')

            if os.path.exists(baseline_file):
                logging.info(f"Loading cached baseline from {baseline_file}")
                stats = np.load(baseline_file, allow_pickle=True).item()
                ctrl_mean, ctrl_std = stats['mean'], stats['std']
            else:
                # Generate baseline logic
                control_files = []
                for root, dirs, files in os.walk(control_dir):
                    for file in files:
                        if file.endswith('.h5'): control_files.append(os.path.join(root, file))

                if not control_files: raise ValueError(f"No .h5 files found in {control_dir}.")

                control_metrics = []
                processed_subjects = set()

                for i, c_path in enumerate(control_files):
                    if os.path.normpath(c_path) == os.path.normpath(patient_h5_path): continue

                    parent_folder = os.path.dirname(c_path)
                    if parent_folder in processed_subjects:
                        continue

                    try:
                        # The helper function auto-merges AP and PA if it finds the partner
                        c_metric = self._load_and_compute_metric(c_path, pd, h5py, np)
                        control_metrics.append(c_metric)
                        processed_subjects.add(parent_folder)
                    except Exception as e:
                        logging.warning(f"Skipping file {c_path}: {e}")

                control_metrics = np.array(control_metrics)
                ctrl_mean = np.mean(control_metrics, axis=0)
                ctrl_std = np.std(control_metrics, axis=0)

                np.save(baseline_file, {'mean': ctrl_mean, 'std': ctrl_std})
                logging.info(f"*** GENERATED NEW BASELINE ***")
                logging.info(f"Saved to: {baseline_file}")
                logging.info(f"Total Unique Controls Processed: {len(control_metrics)}")

            # Calculate Z-Scores
            ctrl_std[ctrl_std == 0] = 1e-9
            z_scores = (patient_metric - ctrl_mean) / ctrl_std
            z_scores = np.nan_to_num(z_scores)

            max_z = np.max(z_scores)

            # Create Heatmap in MNI Space
            heatmap_data = np.zeros_like(atlas_data, dtype=np.float32)
            unique_labels = np.unique(atlas_data)

            # TODO: Ensure label indices (1-434) match the z_score array indices (0-433).
            # If atlas has labels > 434, this will silently skip them. Consider adding a warning.
            for label in unique_labels:
                if label == 0: continue
                if label <= len(z_scores):
                    score = z_scores[int(label) - 1]
                    heatmap_data[atlas_data == label] = score

            # Create MNI Heatmap Node
            heatmap_mni = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Heatmap_MNI_Space")
            heatmap_mni.CreateDefaultDisplayNodes()
            heatmap_mni.CopyOrientation(atlas_node)
            slicer.util.updateVolumeFromArray(heatmap_mni, heatmap_data)

            # Hide intermediate MNI heatmap
            heatmap_mni.GetDisplayNode().SetVisibility(0)

            # ---------------------------------------------------------------
            # 3. REGISTER & WARP (INVERSE: MNI -> PATIENT)
            # ---------------------------------------------------------------
            # Step A: Register Patient -> MNI to get the transform
            # NOTE: Currently the registration fails (poor alignment), the heatmap is offset and not displaying correctly.
            # Check output log for BRAINSFit errors.
            patient_to_mni_transform = self.registerPatientToMNI(mni_node, anatomicalVolume)

            # Step B: Invert Transform (MNI -> Patient)
            patient_to_mni_transform.Inverse()
            patient_to_mni_transform.SetName("MNI_To_Patient_Transform")

            # Step C: Harden Transform on Heatmap
            heatmap_native = slicer.modules.volumes.logic().CloneVolume(heatmap_mni, "DBS_Target_Heatmap_Native")
            heatmap_native.SetAndObserveTransformNodeID(patient_to_mni_transform.GetID())
            slicer.vtkSlicerTransformLogic().hardenTransform(heatmap_native)

            # ---------------------------------------------------------------
            # 4. VISUALIZATION & TARGETING
            # ---------------------------------------------------------------

            # A. Show Patient Anatomy as Background
            slicer.util.setSliceViewerLayers(background=anatomicalVolume, foreground=heatmap_native)

            # B. Style Heatmap
            disp = heatmap_native.GetDisplayNode()
            disp.AutoWindowLevelOn()
            colorNode = slicer.util.getNode("ColdToHotRainbow") or slicer.util.getNode("Rainbow")
            if colorNode: disp.SetAndObserveColorNodeID(colorNode.GetID())
            disp.SetOpacity(0.6)

            # C. Find Max Z in NATIVE Space
            native_data = slicer.util.arrayFromVolume(heatmap_native)
            max_indices = np.unravel_index(np.argmax(native_data), native_data.shape)

            # Convert IJK (Native) -> RAS (Native Scanner Space)
            ijk_point = [max_indices[2], max_indices[1], max_indices[0]]
            ijkToRas = vtk.vtkMatrix4x4()
            heatmap_native.GetIJKToRASMatrix(ijkToRas)

            ras_homog = ijkToRas.MultiplyPoint([ijk_point[0], ijk_point[1], ijk_point[2], 1.0])
            ras_point = ras_homog[:3]

            # D. Place Fiducial
            markups_node_name = "Candidate Target"
            markups_node = None

            try:
                markups_node = slicer.util.getNode(markups_node_name)
            except:
                pass

            if not markups_node:
                markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", markups_node_name)

            markups_node.RemoveAllControlPoints()
            markups_node.AddControlPoint(ras_point[0], ras_point[1], ras_point[2])
            markups_node.SetNthControlPointLabel(0, f"Max Z: {max_z:.2f}")

            # E. Visual Skull Strip (On Native Anatomy)
            disp_anat = anatomicalVolume.GetDisplayNode()
            disp_anat.AutoWindowLevelOn()
            disp_anat.SetApplyThreshold(1)
            anat_arr = slicer.util.arrayFromVolume(anatomicalVolume)
            # NOTE: Adjust this 0.05 threshold if the brain looks too eroded or the skull is still visible.
            disp_anat.SetThreshold(np.max(anat_arr) * 0.05, np.max(anat_arr))

            # F. Center View
            layoutManager = slicer.app.layoutManager()
            threeDWidget = layoutManager.threeDWidget(0)
            threeDView = threeDWidget.threeDView()
            threeDView.resetFocalPoint()

            for sliceNode in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
                sliceNode.JumpSlice(ras_point[0], ras_point[1], ras_point[2])

            logging.info('Processing completed')

        except Exception as e:
            logging.error(f"Error in logic: {e}")
            raise e
        finally:
            slicer.app.restoreOverrideCursor()

    def _load_and_compute_metric(self, h5_path, pd, h5py, np):
        """
        Helper to load .h5 file and calculate Global Connectivity Strength.
        AUTO-MERGE: Looks for matching AP/PA file and averages connectivity.
        """

        def get_data(path):
            try:
                df = pd.read_hdf(path)
                return df.values
            except:
                with h5py.File(path, 'r') as f:
                    key = list(f.keys())[0]
                    return f[key][:]

        # Load Primary File
        data1 = get_data(h5_path)
        # TODO: Validate that data1 shape matches expected region count (434).
        # If not, the atlas mapping will be misaligned.
        if data1.shape[0] == 434: data1 = data1.T
        fc1 = pd.DataFrame(data1).corr().values

        # Look for Partner File (AP <-> PA)
        folder, filename = os.path.split(h5_path)
        if 'restAP' in filename:
            partner_name = filename.replace('restAP', 'restPA')
        elif 'restPA' in filename:
            partner_name = filename.replace('restPA', 'restAP')
        else:
            partner_name = None

        if partner_name:
            partner_path = os.path.join(folder, partner_name)
            if os.path.exists(partner_path):
                try:
                    data2 = get_data(partner_path)
                    if data2.shape[0] == 434: data2 = data2.T
                    fc2 = pd.DataFrame(data2).corr().values

                    # Average the Correlation Matrices
                    fc_matrix = (fc1 + fc2) / 2.0
                except Exception as e:
                    logging.warning(f"   Failed to load partner {partner_name}: {e}. Using single scan.")
                    fc_matrix = fc1
            else:
                fc_matrix = fc1
        else:
            fc_matrix = fc1

        # Metric: Mean Absolute Connectivity (Global Strength)
        metric = np.mean(np.abs(fc_matrix), axis=1)
        return metric


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
        self.test_DBSTargetLocator_Workflow()

    def test_DBSTargetLocator_Workflow(self):
        """
        Test that the basic analysis workflow runs.
        """
        self.delayDisplay("Starting workflow test")

        import numpy as np
        import pandas as pd
        import nibabel as nib

        tempDir = slicer.app.temporaryPath
        patient_path = os.path.join(tempDir, "test_patient.h5")

        resource_dir = os.path.join(tempDir, 'Resources')
        control_dir = os.path.join(resource_dir, 'HealthyControls')
        atlas_dir = os.path.join(resource_dir, 'Atlas')

        os.makedirs(control_dir, exist_ok=True)
        os.makedirs(atlas_dir, exist_ok=True)

        # Note: These dummy files are placeholders.
        # Real tests should use small, valid NIfTI and H5 samples included in the repo.
        atlas_path = os.path.join(atlas_dir, 'Master_DBS_Atlas.nii.gz')
        mni_path = os.path.join(atlas_dir, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')

        n_regions = 434
        n_time = 10

        p_data = np.random.rand(n_time, n_regions)
        pd.DataFrame(p_data).to_hdf(patient_path, key='data')

        for i in range(3):
            c_data = np.random.rand(n_time, n_regions)
            pd.DataFrame(c_data).to_hdf(os.path.join(control_dir, f"ctrl_{i}.h5"), key='data')

        atlas_data = np.random.randint(0, n_regions + 1, (20, 20, 20)).astype(np.int32)
        affine = np.eye(4)
        img = nib.Nifti1Image(atlas_data, affine)
        nib.save(img, atlas_path)

        mni_data = np.random.randint(0, 255, (20, 20, 20)).astype(np.uint8)
        mni_img = nib.Nifti1Image(mni_data, affine)
        nib.save(mni_img, mni_path)

        anat_data = np.random.randint(0, 255, (20, 20, 20)).astype(np.uint8)
        anatomicalVolume = slicer.util.addVolumeFromArray(anat_data)

        logic = DBSTargetLocatorLogic()

        original_get_resource = logic.get_module_resource_path
        logic.get_module_resource_path = lambda: resource_dir

        try:
            logic.analyzeFMRI(patient_path, anatomicalVolume)
            self.delayDisplay("Test passed!")
        except Exception as e:
            self.delayDisplay(f"Test failed: {e}")
            raise e
        finally:
            logic.get_module_resource_path = original_get_resource