## 1.  Dataset Overview
### 1.1.  Dataset Summary

| Parameter                        | Details                          |
| -------------------------------- | -------------------------------- |
| **Athletes**                     | 13                               |
| **Sessions**                     | 38                               |
| **Repetitions per Session/File** | 5                                |
| **Total Repetitions**            | 180                              |
| **IMU Sensors**                  | 26                               |
| **Sampling Frequency**           | 100 Hz                           |
| **Synchronization**              | Timestamp synchronized           |
| **No. of Sensors**               | 26 IMU (Accelerometer,Gyroscope) |
### 1.2.  Athlete Metadata
The dataset contains additional athlete-specific information:
- Athlete information
- Sex
- Age
- Height
- Body mass
- Weight lifted
- Training experience / skill level
- Achievements / performance background
### 1.3.  Raw Data
- Accelerometer
- Gyroscope
- Timestamp-synchronized measurements
- Athlete Metadata
### 1.4.  Processed Data
- Filtered IMU attitude/orientation
- Knee angle estimations
- Elbow angle estimations

### 1.5.  Use Cases of Data
- Weightlifting technique analysis
- Movement analysis
- Biomechanical research
- Machine learning
- Human activity recognition
- Injury-risk research
- Sensor fusion / IMU research

**13 Athletes → 38 Sessions → 5 Repetitions per Session → 190 Total Repetitions**
**26 IMUs × 100 Hz sampling frequency**
The dataset therefore contains both **raw synchronized sensor measurements** and **processed biomechanical variables**, together with **athlete and performance metadata**.

## 2.  Journals
| Journal         | Publisher                                             | SJR 2025 | Quartile | H-Index | ISSN     | Coverage  | Subject Areas & Categories                                                                                                                                                                                                                 | Category Type   | Best Fit / Relevance                                                                                     |
| --------------- | ----------------------------------------------------- | -------- | -------- | ------- | -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- | -------------------------------------------------------------------------------------------------------- |
| Scientific Data | Nature Research                                       | 1.907    | Q1       | 165     | 20524463 | 2014–2026 | Computer Science: Computer Science Applications, Information Systems • Decision Sciences: Statistics, Probability and Uncertainty • Mathematics: Statistics and Probability • Social Sciences: Education, Library and Information Sciences | Dataset-Focused | 5/5 Dataset publication — Specifically designed for publishing valuable datasets and their documentation |
| Data in Brief   | Elsevier Inc.                                         | 0.606    | Q1       | 69      | 23523409 | 2014–2026 | Multidisciplinary:Multidisciplinary • Social Sciences: Education                                                                                                                                                                           | Dataset-Focused | 5/5 Dataset publication — Strong option for a dedicated dataset/data descriptor paper                    |
| Data            | Multidisciplinary Digital Publishing Institute (MDPI) | 0.538    | Q2       | 49      | 23065729 | 2016-2026 | Computer Science<br>    Computer Science Applications<br>    Information Systems<br>Decision Sciences<br>    Information Systems and Management                                                                                            | Dataset-Focused |                                                                                                          |


## 3.  Scientific Data
**Scientific Data** mainly publishes papers that describe **useful research datasets** and explain how those datasets were **collected, processed, checked,** and made available for other researchers to use.

The main paper type is a **Data Descriptor**. It should explain
- How the data were collected
- What the dataset contains
- How the data were processed
- How the quality of the data was checked/validated
- How other researchers can reuse the dataset

Importantly, the paper **does not need to prove a scientific hypothesis using the data**. Scientific Data specifically says it does not publish traditional research papers where data are mainly used to validate hypotheses.

### 3.1.  Structure
- Title
- Author list (does not need a heading)
- Abstract
- Background & Summary
- Methods
- Data Records
- Data Overview (optional)
- Technical Validation
- Usage Notes (optional)
- Data Availability
- Code Availability
- References
- Author Contributions
- Competing Interests
- Acknowledgements (optional)
- Funding
- Ethics statement*

### 3.2.  Deposit the dataset
- First Round anonymous download link or public (recommended)
- Second Round public
- Generalist repository
	- [Dryad Digital Repository](https://datadryad.org/stash)
	- [figshare](https://figshare.com/)
	- [Harvard Dataverse](https://dataverse.harvard.edu/)
	- [Open Science Framework](https://osf.io/)
	- [Science Data Bank](https://www.scidb.cn/en)
	- [Zenodo](https://zenodo.org/)
- Ensure long-term persistence and preservation of datasets
- Make data available for peer review
- Provide stable persistent identifiers for submitted datasets
- CC-BY - Attribution: anyone can use it, but they must credit you as the author or
- CC0 - Public Domain: anyone can use it for any reason without even asking
- If we publich videos as well need to check wether they are sensitive human data 
- do not require or encourage the use of article templates
- Titles must not exceed 110 characters
- Abstract should not exceed 170 word
- If you wish to name more than one first author please use the additional footnote such as "These authors contributed equally".
### 3.3.  Repository Feature Comparison

| **Repository**        | **Free Storage Limit**               | **Data Curation Level**     | **License Options**            | **Anonymous Peer-Review Link**  | **Primary Strengths**                                            |
| --------------------- | ------------------------------------ | --------------------------- | ------------------------------ | ------------------------------- | ---------------------------------------------------------------- |
| **Zenodo**            | 50 GB / dataset                      | Automated / Light check     | CC0, CC-BY, Custom             | Yes (Secret link)               | High capacity, CERN-backed, GitHub integration                   |
| **Figshare**          | 20 GB total (5 GB/file)              | Self-curated                | CC0, CC-BY, etc.               | Yes (Private share link)        | Inline file previews, simple interface                           |
| **Harvard Dataverse** | 1 TB total per dataset               | Institutional / Admin check | Default CC0 (Custom supported) | Yes (Private URL / Draft mode)  | Deep metadata schemas, tabular data tools                        |
| **Dryad**             | 300 GB ($120+ fee unless affiliated) | Full Human Curation         | Strictly CC0                   | Yes (Reviewer link)             | Professional data curation, high FAIR compliance                 |
| **OSF**               | 5 GB/component (50 GB total)         | Self-curated                | CC0, CC-BY, MIT, etc.          | Yes (Anonymized view-only link) | Modular project organization, cloud integrations                 |
| **SciDB**             | ~50 GB / dataset                     | Staff / Automated check     | CC0, CC-BY 4.0                 | Yes (Reviewer access)           | Direct Springer Nature/Scientific Data integration               |
| **Mendeley Data***    | 10 GB / dataset                      | Self-curated                | CC0, CC-BY, CC-BY-NC, etc.     | Yes (Private share link)        | Elsevier ecosystem integration, versioning, directory structures |

- **Zenodo:** Large time-series datasets, raw motion-capture/IMU arrays, sensor logs, and codebases requiring up to 50 GB free storage.
- **Figshare:** Supplementary material, figures, media files, and small-to-medium datasets needing fast web-based previewing.
- **Harvard Dataverse:** Social science, clinical, or complex multi-variable experiments requiring custom metadata terms and variables.
- **Dryad:** High-impact life sciences and ecology data where dedicated data curators review files for long-term usability.
- **Open Science Framework (OSF):** Collaborative, ongoing research projects requiring integration across GitHub, Google Drive, and pre-registrations.
- **Science Data Bank (SciDB):** Multidisciplinary scientific data paired with Nature/Springer publications, particularly for Asian and global repository indexing.
- <font color="#938953">Mendeley Data: Version-controlled datasets requiring hierarchical folder structures, particularly suited for those wanting standard open-access archiving up to 10 GB or utilizing the Elsevier ecosystem.</font>
		<font color="#938953">	Not in recomnded list</font>

 **First Choice: Zenodo**
High-frequency, 26-sensor time-series logs generated across 180 total repetitions will accumulate significant file size. Zenodo’s free 50 GB limit per deposit accommodates multi-gigabyte ZIP/CSV file trees without storage fees, whereas alternatives like Mendeley Data (capped at 10 GB per dataset) might require awkwardly splitting the raw kinematic data across multiple deposits if it exceeds that threshold.
### 3.4.  Papers Data in Scientific Data

#### 3.4.1.  Database covering the prayer movements which were not available previously
- Background & Summary
	- Basic Summary
	- Comparison of other datasets
- Methods
	- Participants.
	- Experimental set-up and equipment.
	- Utilized motion capture systems.
	- IMU Based Wearable MOCAP System.
	- Study design
	- Procedure
	- List of activities and their definitions in the database
- Data Records
- Technical Validation

#### 3.4.2.  A human lower-limb biomechanics Data Descriptor and wearable sensors dataset during cyclic and non-cyclic activities
- Background & Summary
	- Compare with other datasets
- Methods
	- Participants.
	- Performed tasks
	- Motion capture
	- Ground reaction forces
	- Joint angles and velocities
	- Joint moments and powers
	- Electromyography.
	- IMUs.
	- <font color="#f79646">Virtual inertial measurement units (IMUs).Filtered kinematics in OpenSim were used to calculatetransformation matrices to transform segment positions into global frame coordinates or each individual time step.</font>
	- <font color="#f79646">Virtual insoles</font>
	- Summary data visualizations
- Data Records
- Technical Validation - just explanations no maths or equations
	- Data collection
	- Data processing 
	- Comparison to public datasets
<font color="#e36c09">There is interesting plot for cyclic events check that</font>

#### 3.4.3.  A dataset for fatigue estimation during shoulder internal and external rotation movements using wearables
- Background & Summary
- Methods
	- Participants.
	- Experimental protocol
	- Anthropometric measurements.
	-  Other Measurements
- Technical Validation
	- Limitations. - Data clean up, magnetic field inteference
- Usage Note
#### 3.4.4.  A Full-Body IMU-Based Motion Data Descriptor Dataset of Daily Tasks by Older and Younger Adults
- Background & Summary
- Methods
	- Participants.
	- Wearable IMU-based Motion Capture (MoCap)
	- Movement tasks
	- <font color="#f79646">Calibration</font>
	- Data Segmentation
	- Acquisition setup and procedure
	- Data pre-processing
- Data Records
- <font color="#f79646">Technical Validation _ THIS IS NICE CHECK</font>
- Usage Notes
#### 3.4.5.  GAITEX: Human motion dataset Data Descriptor of impaired gait and rehabilitation exercises using inertial and optical sensors
- Background & Summary
- Methods
	- Participants and ethical requirements.
	- Acquisition Setup
	- Acquisition protocol.
	- Inverse Kinematics using IMU Data and Musculoskeletal Models.


## 4.  Data in Brief
**Data in Brief** is also focused on publishing **research datasets**, but its approach is to provide a **short, clear and easy-to-understand data article** explaining the dataset and making it accessible to other researchers.

It is multidisciplinary and open access. Its main article type is the **data article**, which describes and provides access to research data.

The journal wants data that:
- Were collected using a scientific method
- Have value to the research community
- Can be shared and reused
- Improve reproducibility and discoverability
It also supports the **FAIR data principles**.

1. Title page
2. Abstract
3. Keywords, Graphical abstract
4. Units, classifications codes and nomenclature
5. Research data
6. Data linking



### 4.1.  Papers Data in Brief
#### 4.1.1.  A dataset for the development and optimization of fall detection algorithms based on wearable sensors
- 8 subjects while simulating 13 types of falls and 5 types of Activities of Daily Living (ADL), each repeated 3 times.
- 13 types - 4 backeard, 4 forward, 2 right, 2 left, 1 syncope
- Data were acquired using a MARG sensor, a wearable multisensory device tied to the subject's waist, that recorded time-variations of the subject's ac- celeration and orientation (expressed through the yaw, pitch and roll angles). 
- These data can be useful in the development and test of algorithms to automatically identify and classify fall events.

1. **Data**
The proposed dataset is the only one that provides a complete set of data detected through a MARG
sensor....
explain data structure and files and folders

2. **Experimental design, materials and methods**
Explain people and how experiment was done

3. **Acknowledgments**

#### 4.1.2.  Dataset of 3D gait analysis in typically developing children walking at three different speeds on an instrumented treadmill in virtual reality

- In this article, gait data of typically developing (TD) children (24 boys/31 girls, mean (95%  onﬁdence interval) age 9.38 (8.51 – 10.25) years, body mass 35.67 (31.40 – 39.94) kg, leg length 0.73 (0.70 – 0.76) m, and height 1.41 (1.35 – 1.46) m walking at different walking speeds is shared publicly
- For clinical application,
- Gait analysis was performed during treadmill walking in a virtual environment using the Computer Assisted Rehabilitation Environment (CAREN). The human body lower limbmodel with trunk markers (HBM2) was used as biomechanical model.
1. **Objective**
2. **Data Description**
3. **Experimental Design, Materials and Methods**
4. **Ethics Statements**

#### 4.1.3.  Multimodal human motion dataset of 3D anatomical landmarks and pose keypoints
- ataset that takes 2D and 3D human pose keypoints estimated from images and relates them to the location of 3D anatomical landmarks
- The dataset contains 51,051 poses obtained from 71 persons in A-Pose while performing 7 movements (walking, running, squatting, and four types of jumping).
1. Value of the Data
2. Background
3. Data Description
4. Experimental Design, Materials and Methods
5. Limitations
6. Ethics Statement

#### 4.1.4.  Dataset of running kinematics, kinetics and muscle activation at different speeds, surface gradients, cadences and with forward trunk lean

- Dataset of running kinematics, kinetics and muscle activation at different speeds, surface gradients, cadences and with forward trunk lean of various performancelevels (untrained to Olympic level athlete)
- <font color="#e36c09">Both raw data (.C3D ﬁles that include the 3D positions of the markers, ground reaction forces, and surface muscle activation) and processed data (joint angles, joint moments, muscle forces, joint contact forces, all analysed using OpenSim 3.3 musculoskeletal modelling software) are available.</font>
- The data can be re-used for various purposes including model validation studies, studies that aim to investigate biomechanical outcomes in speciﬁc running condition
1. Same sub sections 

#### 4.1.5.  IMU50: a high-frequency multi-modal wrist-worn IMU dataset of 50 subjects over 5 continuous days for supervised and self-supervised learning in free-living conditions
- a multiday, high-resolution inertial measurement unit (IMU) dataset collected from 50 healthy volunteers using the Actigraph Leap wrist-worn device. Each participant was monitored continuously for ﬁve consecutive days and nights, yielding an average of approximately 136 h of raw sensor recordings per subject.
- The dataset includes three-axis ac-celerometer signals, three-axis gyroscope signals sampled at 128 Hz, double-channel photoplethysmogram (PPG) sampledat 25 Hz,
- <font color="#e36c09">Compare with other available datasets</font>
- <font color="#e36c09">Included Data Distributions and stats</font>


