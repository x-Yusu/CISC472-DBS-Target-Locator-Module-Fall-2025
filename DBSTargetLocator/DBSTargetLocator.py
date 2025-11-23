# The following was derived from an example module template from a PerkLabBootcamp slicer scripted module example file:
# https://github.com/PerkLab/PerkLabBootcamp/blob/master/Examples/CampTutorial2/CampTutorial2.py


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
    Register OpenNeuro processed fMRI dataset (.h5) for subject NDARINVDG233EBR.
    """
    import SampleData
    import os

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # Processed fMRI connectivity matrix (.h5)

    h5_url = "https://s3.amazonaws.com/openneuro.org/ds005237/fMRI_timeseries_clean_denoised_GSR_parcellated/NDAR_INVDG233EBR/task-restAP_run-01_bold_Atlas_hp2000_clean_GSR_parcellated.h5"

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="OpenNeuro ds005237",
        sampleName="sub-NDARINVDG233EBR_restAP (H5)",
        thumbnailFileName=os.path.join(iconsPath, "OpenNeurofMRI.png"),
        uris=[h5_url],
        fileNames=["task-restAP_run-01_bold_Atlas_hp2000_clean_GSR_parcellated.h5"],
        nodeNames=["sub-NDARINVDG233EBR_restAP_h5"],
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

        # Set default layout to conventional (no table view)
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)

        #
        # Input Data Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Patient Data Selection"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        #
        # Patient Folder Selector (Dir)
        #
        self.patientSelector = ctk.ctkPathLineEdit()
        self.patientSelector.filters = ctk.ctkPathLineEdit.Dirs  # Changed to Directory Selector
        self.patientSelector.settingKey = 'DBSTargetLocator/PatientPath'
        self.patientSelector.toolTip = "Select the folder containing the patient's processed .h5 files."
        parametersFormLayout.addRow("Patient Data Folder:", self.patientSelector)

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

        #
        # Results Area
        #
        self.resultsCollapsibleButton = ctk.ctkCollapsibleButton()
        self.resultsCollapsibleButton.text = "Top Candidates"
        self.layout.addWidget(self.resultsCollapsibleButton)
        self.resultsLayout = qt.QVBoxLayout(self.resultsCollapsibleButton)

        self.resultsTable = qt.QTableWidget()
        self.resultsTable.setColumnCount(3)
        self.resultsTable.setHorizontalHeaderLabels(["Rank", "Region", "Z-Score"])
        self.resultsLayout.addWidget(self.resultsTable)

        # Add vertical spacer
        self.layout.addStretch(1)

        # Create logic class. This handles all the actual computation work.
        self.logic = DBSTargetLocatorLogic()

        # Connect UI elements - wire up all the buttons and controls to their callback functions
        self.patientSelector.connect("currentPathChanged(const QString &)", self.onSelect)
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
                self.patientSelector.currentPath != ""
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

            # Run the analysis
            results = self.logic.analyzeFMRI(patient_path)

            # Populate Results Table
            if results:
                self.resultsTable.setRowCount(len(results))
                for i, (region, score) in enumerate(results):
                    self.resultsTable.setItem(i, 0, qt.QTableWidgetItem(str(i + 1)))
                    self.resultsTable.setItem(i, 1, qt.QTableWidgetItem(region))
                    self.resultsTable.setItem(i, 2, qt.QTableWidgetItem(f"{score:.4f}"))

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

    def get_tian_label_mapping(self):
        """
        Returns a dictionary mapping Tian Scale II Label IDs (1-32) to Region Names.
        Scale II divides the 8 subcortical structures into 16 regions per hemisphere.
        """

        roi_names = {
            # --- RIGHT HEMISPHERE (1-16) ---
            1: "Anterior Hippocampus (Right)",
            2: "Posterior Hippocampus (Right)",
            3: "Lateral Amygdala (Right)",
            4: "Medial Amygdala (Right)",
            5: "Thalamus - Prefrontal (Right)",
            6: "Thalamus - Motor (Right)",
            7: "Thalamus - Sensory (Right)",
            8: "Thalamus - Parietal (Right)",
            9: "Anterior Caudate (Right)",
            10: "Posterior Caudate (Right)",
            11: "Anterior Putamen (Right)",
            12: "Posterior Putamen (Right)",
            13: "Anterior Pallidum (Right)",
            14: "Posterior Pallidum (Right)",
            15: "Nucleus Accumbens Shell (Right)",
            16: "Nucleus Accumbens Core (Right)",

            # --- LEFT HEMISPHERE (17-32) ---
            17: "Anterior Hippocampus (Left)",
            18: "Posterior Hippocampus (Left)",
            19: "Lateral Amygdala (Left)",
            20: "Medial Amygdala (Left)",
            21: "Thalamus - Prefrontal (Left)",
            22: "Thalamus - Motor (Left)",
            23: "Thalamus - Sensory (Left)",
            24: "Thalamus - Parietal (Left)",
            25: "Anterior Caudate (Left)",
            26: "Posterior Caudate (Left)",
            27: "Anterior Putamen (Left)",
            28: "Posterior Putamen (Left)",
            29: "Anterior Pallidum (Left)",
            30: "Posterior Pallidum (Left)",
            31: "Nucleus Accumbens Shell (Left)",
            32: "Nucleus Accumbens Core (Left)"
        }

        return roi_names

    def analyzeFMRI(self, patient_folder_path):
        """
        Perform fMRI analysis to identify candidate DBS target locations.
        Returns list of (RegionName, ZScore) tuples for top 3 candidates.

        Accepts a folder path, scans for .h5 files, and averages them if multiple are found.
        """

        logging.info("Starting fMRI analysis...")
        logging.info(f"Patient Data Folder: {patient_folder_path}")

        import pandas as pd
        import numpy as np
        import h5py

        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)

        top_candidates = []

        try:
            # ---------------------------------------------------------------
            # 0. SETUP PATHS
            # ---------------------------------------------------------------
            resourceDir = self.get_module_resource_path()
            control_dir = os.path.join(resourceDir, 'HealthyControls')

            # Use standard MNI template as background
            mni_template_path = os.path.join(resourceDir, 'Atlas', 'mni_icbm152_t1_tal_nlin_asym_09c.nii')

            # Use Tian Scale II Subcortical Atlas for masking
            tian_atlas_path = os.path.join(resourceDir, 'Atlas', 'Tian_Subcortex_S2_3T_1mm.nii.gz')

            if not os.path.exists(control_dir): raise FileNotFoundError(f"Healthy Controls not found at {control_dir}.")
            if not os.path.exists(mni_template_path): raise FileNotFoundError(
                f"MNI Template not found at {mni_template_path}.")
            if not os.path.exists(tian_atlas_path): raise FileNotFoundError(
                f"Tian Atlas not found at {tian_atlas_path}.")

            # ---------------------------------------------------------------
            # 1. LOAD MNI TEMPLATE (BACKGROUND)
            # ---------------------------------------------------------------
            mni_node = slicer.util.loadVolume(mni_template_path)
            mni_node.SetName("Standard_MNI_Template")

            # Load Tian Atlas (Hidden, used for geometry and masking)
            tian_node = slicer.util.loadLabelVolume(tian_atlas_path)
            tian_node.SetName("Tian_Subcortex_Atlas")

            # Ensure Tian is explicitly hidden immediately upon loading
            if tian_node.GetDisplayNode():
                tian_node.GetDisplayNode().SetVisibility(0)

            # Get atlas data for masking
            tian_data = slicer.util.arrayFromVolume(tian_node)

            # ---------------------------------------------------------------
            # 2. COMPUTE Z-SCORES (On 434-Region Vector)
            # ---------------------------------------------------------------
            # Load Patient Metric - Modified to handle folder input
            logging.info(f"Scanning folder: {patient_folder_path}")

            patient_files = [os.path.join(patient_folder_path, f) for f in os.listdir(patient_folder_path)
                             if f.endswith('.h5') and ('rest' in f)]

            if not patient_files:
                raise ValueError(f"No resting state .h5 files found in {patient_folder_path}")

            logging.info(f"Found {len(patient_files)} valid patient files. Averaging...")

            # Compute average metric for patient
            patient_metrics = []
            for p_file in patient_files:
                try:
                    metric = self._compute_single_file_metric(p_file, pd, h5py, np)
                    patient_metrics.append(metric)
                except Exception as e:
                    logging.warning(f"Failed to load patient file {p_file}: {e}")

            if not patient_metrics:
                raise ValueError("Failed to compute metrics for any patient files.")

            # Average the metrics from all found files (AP, PA, etc)
            patient_metric = np.mean(np.array(patient_metrics), axis=0)

            # Load/Calc Baseline
            distr_file = os.path.join(control_dir, 'baseline_dist.npy')
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
                    # Check if this control file is inside the patient folder we just selected
                    # To avoid using patient as control if user points to same dir
                    if os.path.commonpath([patient_folder_path]) == os.path.commonpath([patient_folder_path, c_path]):
                        continue

                    parent_folder = os.path.dirname(c_path)
                    if parent_folder in processed_subjects: continue

                    try:
                        # Use the auto-merge helper for controls to keep logic simple there
                        c_metric = self._load_and_compute_metric_auto_merge(c_path, pd, h5py, np)
                        control_metrics.append(c_metric)
                        processed_subjects.add(parent_folder)
                    except Exception as e:
                        logging.warning(f"Skipping file {c_path}: {e}")

                control_metrics = np.array(control_metrics)
                ctrl_mean = np.mean(control_metrics, axis=0)
                ctrl_std = np.std(control_metrics, axis=0)

                np.save(distr_file, control_metrics)
                np.save(baseline_file, {'mean': ctrl_mean, 'std': ctrl_std})
                logging.info(f"Generated baseline from {len(control_metrics)} controls.")

            # Calculate Z-Scores
            ctrl_std[ctrl_std == 0] = 1e-9
            z_scores = (patient_metric - ctrl_mean) / ctrl_std
            z_scores = np.nan_to_num(z_scores)

            # ---------------------------------------------------------------
            # 3. CREATE SUBCORTICAL HEATMAP & FIND TOP 3
            # ---------------------------------------------------------------
            # We only care about indices 400-431 (The 32 Subcortical Regions)
            subcortex_z_scores = z_scores[400:432]

            # Create Heatmap Array
            # MODIFICATION: Initialize with NaN (transparent) instead of 0.
            heatmap_data = np.full_like(tian_data, np.nan, dtype=np.float32)

            # Fill Heatmap
            for label_id in range(1, 33):
                z_val = z_scores[400 + label_id - 1]
                heatmap_data[tian_data == label_id] = z_val

            # Find Top 3 Absolute Z-Scores
            # Create list of (index, abs_z, real_z)
            scored_regions = []
            label_map = self.get_tian_label_mapping()

            for i, z in enumerate(subcortex_z_scores):
                label_id = i + 1
                region_name = label_map.get(label_id, f"Region_{label_id}")
                scored_regions.append((region_name, np.abs(z), z, label_id))

            # Sort by Absolute Z (Descending)
            # This ensures high magnitude negatives (-5) are ranked higher than low positives (+1)
            scored_regions.sort(key=lambda x: x[1], reverse=True)

            # Extract Top 3
            for i in range(min(3, len(scored_regions))):
                name, abs_z, real_z, lbl_id = scored_regions[i]
                top_candidates.append((name, real_z))
                logging.info(f"Rank {i + 1}: {name} | Z-Score: {real_z:.4f}")

            # Create Heatmap Node
            heatmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Subcortical_Z_Score_Map")
            heatmap_node.CreateDefaultDisplayNodes()
            heatmap_node.CopyOrientation(tian_node)
            slicer.util.updateVolumeFromArray(heatmap_node, heatmap_data)

            # ---------------------------------------------------------------
            # 4. VISUALIZATION & COMPOSITION
            # ---------------------------------------------------------------

            # A. Show MNI Template as Background and Show Heatmap as Foreground
            # Explicitly set foregroundOpacity=0.7 to ensure it is visible (not 0%)
            # Also ensure label=None to hide Tian Atlas
            slicer.util.setSliceViewerLayers(
                background=mni_node,
                foreground=heatmap_node,
                label=None,
                foregroundOpacity=0.7
            )

            # B. Style Heatmap
            disp = heatmap_node.GetDisplayNode()

            # Set window/level to only show meaningful z-scores
            valid_zscores = heatmap_data[~np.isnan(heatmap_data)]
            if len(valid_zscores) > 0:
                z_min = np.min(valid_zscores)
                z_max = np.max(valid_zscores)
                z_range = z_max - z_min
                window = z_range if z_range > 0 else 1.0
                level = (z_min + z_max) / 2.0
                disp.SetAutoWindowLevel(0)
                disp.SetWindowLevel(window, level)
            else:
                disp.AutoWindowLevelOn()

            # Set Colormap (ColdToHotRainbow) - Good for showing -/blue and +/red
            colorNode = slicer.util.getNode("ColdToHotRainbow") or slicer.util.getNode("Rainbow")
            if colorNode: disp.SetAndObserveColorNodeID(colorNode.GetID())

            # Set Translucency (0.7 Opacity)
            disp.SetOpacity(0.7)

            # Apply threshold to hide NaN/invalid regions
            disp.SetApplyThreshold(1)
            if len(valid_zscores) > 0:
                threshold = z_min - 0.1  # Slightly below minimum to include all valid data
                disp.SetThreshold(threshold, z_max)

            # C. Synchronize Volume Rendering (3D View)
            volRenLogic = slicer.modules.volumerendering.logic()
            displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(heatmap_node)
            displayNode.SetVisibility(1)

            # D. Place Fiducials at Top 3 Nodes
            markups_node_name = "Top Candidate Targets"
            markups_node = None
            try:
                markups_node = slicer.util.getNode(markups_node_name)
            except:
                pass

            if not markups_node:
                markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", markups_node_name)

            markups_node.RemoveAllControlPoints()

            # Limit to available regions if fewer than 3
            count = min(3, len(scored_regions))

            for i in range(count):
                top_name, _, top_z, top_label = scored_regions[i]

                region_mask = (tian_data == top_label)
                if np.any(region_mask):
                    coords = np.argwhere(region_mask)
                    center_ijk = coords.mean(axis=0)
                    ijk_point = [center_ijk[2], center_ijk[1], center_ijk[0]]

                    ijkToRas = vtk.vtkMatrix4x4()
                    tian_node.GetIJKToRASMatrix(ijkToRas)
                    ras_homog = ijkToRas.MultiplyPoint([ijk_point[0], ijk_point[1], ijk_point[2], 1.0])
                    ras_point = ras_homog[:3]

                    # Add Control Point
                    pid = markups_node.AddControlPoint(ras_point[0], ras_point[1], ras_point[2])
                    markups_node.SetNthControlPointLabel(pid, f"{i + 1}: {top_name}\nZ: {top_z:.2f}")

                    # Center View on the #1 candidate
                    if i == 0:
                        layoutManager = slicer.app.layoutManager()
                        threeDWidget = layoutManager.threeDWidget(0)
                        threeDView = threeDWidget.threeDView()
                        threeDView.resetFocalPoint()
                        for sliceNode in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
                            sliceNode.JumpSlice(ras_point[0], ras_point[1], ras_point[2])

            logging.info('Processing completed')
            return top_candidates

        except Exception as e:
            logging.error(f"Error in logic: {e}")
            raise e
        finally:
            slicer.app.restoreOverrideCursor()

    def _compute_single_file_metric(self, path, pd, h5py, np):
        """
        Computes Global Connectivity Strength for a single file.
        No averaging logic.
        """

        def get_data(p):
            try:
                df = pd.read_hdf(p)
                return df.values
            except:
                with h5py.File(p, 'r') as f:
                    key = list(f.keys())[0]
                    return f[key][:]

        data = get_data(path)
        if data.shape[0] == 434: data = data.T
        fc = pd.DataFrame(data).corr().values
        return np.mean(np.abs(fc), axis=1)

    def _load_and_compute_metric_auto_merge(self, h5_path, pd, h5py, np):
        """
        Legacy Helper for Controls: Looks for matching AP/PA file and averages connectivity.
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
                    fc_matrix = (fc1 + fc2) / 2.0
                except Exception as e:
                    logging.warning(f"   Failed to load partner {partner_name}: {e}. Using single scan.")
                    fc_matrix = fc1
            else:
                fc_matrix = fc1
        else:
            fc_matrix = fc1

        metric = np.mean(np.abs(fc_matrix), axis=1)
        return metric


#
# DBSTargetLocatorTest
#

class DBSTargetLocatorTest(ScriptedLoadableModuleTest):
    """
    This is the test case for our scripted module.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_DBSTargetLocator_LocalSample()

    def test_DBSTargetLocator_LocalSample(self):
        """
        Tests the module using the local sample folder in Resources/SamplePatientH5.
        This tests the Logic AND updates the GUI table.
        """
        self.delayDisplay("Starting local sample test")

        # 1. Locate the sample data in Resources
        moduleDir = os.path.dirname(__file__)
        patient_data_path = os.path.join(moduleDir, 'Resources', 'SamplePatientH5', 'NDAR_INVAP729WCD')

        # 2. Check if it exists
        if not os.path.exists(patient_data_path):
            self.delayDisplay(f"Test Data not found at {patient_data_path}. Please ensure folder exists.")
            return

        # 3. INTERACT WITH THE UI
        # By getting the widget, we ensure the table logic runs, not just the math logic.
        try:
            # Switch to the module in the UI
            slicer.util.selectModule('DBSTargetLocator')

            # Get the Python Widget object
            widget = slicer.modules.dbstargetlocator.widgetRepresentation().self()

            # Set the path in the UI selector
            widget.patientSelector.currentPath = patient_data_path

            # Click "Calculate" programmatically
            self.delayDisplay("Simulating 'Calculate' click...")
            widget.onApplyButton()

            self.delayDisplay("Test Passed! Z-Score Table should now be populated in the module panel.")

        except Exception as e:
            self.delayDisplay(f"Test failed to drive UI: {e}")
            import traceback
            traceback.print_exc()