# Automated Localization of Candidate Electrode Sites for Deep Brain Stimulation in Depression using fMRI
*** 

### About the Project

Module formatting based off of this [tutorial](https://github.com/Slicer/SlicerProgrammingTutorial/tree/main/Data/Part2/MyFirstExtension/MyFirstModule).\
...
CISC 881/CISC/CMPE 472 - Medical Informatics
November 21st 2025

### Authors
- Gabriel Burrows
- Leopold Ehrlich
- Keiichiro Hayashi
- Robert He
- Amanda Zhu

### Motivation
Deep brain stimulation (DBS) is an established treatment for Parkinson’s as electrode placement is well defined. In treatment of major depressive disorder (MDD) however, the optimal stimulation target zones, parameter of stimulation and stimulation protocols are unclear and vary across patients (Drobisz & Damborská, 2018). An automated localization method for stimulation sites which are likely to respond therapeutically to DBS on a per patient basis could improve efficacy of treatment and in turn clinical outcomes for patients with treatment resistant depression (TRD). This tool would inform clinicians of regions of interest at the individual patient level, providing them with a probabilistic analysis of key brain structures implicated in depressive symptoms, streamlining treatment and surgery planning. This tool would rely on professional opinions before invasive procedures are performed, so introduction of risk associated with its use is mitigated.

### Biomarker Rationale
Resting-state functional Magnetic Resonance Imaging (rs-fMRI) allows for a non-invasive method of identifying patient specific biomarkers. MDD is often called a “neurocircuitry disease” as it involves disruptions in large scale networks, including the cortico-striato-thalamo-cortical (CSTC) loops [1]. Functional hallmarks of MDD include abnormal metabolic activity, with some regions being hyperactive while others are hypoactive. For example, hypoactivity in the subgenual Anterior Cingulate Cortex (sACC) is commonly seen in patients with TRD [1]. Our proposed method of analysis uses the mean BOLD intensity of segmented functional regions as a measure of local cerebral metabolism. This metric can then be compared to a healthy baseline distribution to generate a Z-score indicating statistically significant abnormality in each functional region. We can then inspect the scores in known ROIs associated with MDD, and provide probabilistically ranked target sites for DBS.

### Objectives
1.	Develop an automated module in 3D Slicer for processing fMRI datasets to localize candidate electrode sites for DBS.
2.	Systematically compare frequency spectra characteristics of patient fMRI activity across different regions and compare to baseline activity to highlight depressed regions.
3.	Create a visualization tool for clinicians that lists, ranks and displays in 3D, the most promising sites for DBS based on presence of abnormal low-frequency activity. This tool does not make any decisions independently; it just informs clinical planning and decision making.

### Proposed Methods
1.	Into 3D slicer, import (preprocessed) fMRI data (closed eyes, at resting state) obtained from two study groups: a control group of individuals (without mental illness), and a patient group of individuals diagnosed with major depressive disorder (MDD).
2.	Frequency spectrum analysis is applied to the groups’ 3D time series data, to find patterns of brain activity at the voxel level. Results give the level of activity at each voxel in the fMRI per individual.
3.	Comparisons between group results are performed either through direct statistical comparison or using machine learning models. This step highlights regions in the brain where altered activity is associated with depressive states.
4.	Anatomical data accompanying the fMRI scans will be used to spatially display the individual patient’s brain in 3D slicer. Patient-specific maps of altered activity will be visualized within 3D slicer, scaled to the patient’s brain. 
5.	Regions of interest (ROI) are suggested to clinicians to serve as potential clinical targets for deep brain stimulation therapy for patients with MDD.

### References
[1] D. Drobisz and A. Damborská, “Deep brain stimulation targets for treating depression,” Behavioural Brain Research, vol. 359, pp. 266–273, Nov. 2018, doi: 10.1016/j.bbr.2018.11.004. 

[2] M. Figee, P. Riva-Posse, K. S. Choi, L. Bederson, H. S. Mayberg, and B. H. Kopell, “Deep brain stimulation for depression,” Neurotherapeutics, vol. 19, no. 4, pp. 1229–1245, Jul. 2022, doi: 10.1007/s13311-022-01270-3. 

[3] M. Roet, J. Boonstra, E. Sahin, A. E. P. Mulders, A. F. G. Leentjens, and A. Jahanshahi, “Deep Brain Stimulation for Treatment-Resistant Depression: towards a more personalized treatment approach,” Journal of Clinical Medicine, vol. 9, no. 9, p. 2729, Aug. 2020, doi: 10.3390/jcm9092729. 