# M19 - More training datasets for Gym Pose Lab

**Status:** discovery and evaluation plan only. The public website currently
supports exactly Squat, Overhead Press (OHP), and Barbell Row with two
original-source RF models each. Additional exercises are **not** yet trained,
validated or offered as scoring options.

The current six models were trained on task-specific **Fitness-AQA** source
error labels. A separate exercise recognition dataset cannot automatically
serve as a labeled form-error detector. Dataset licenses and distribution
terms must be respected before training and deployment.

## Ranked dataset candidates (verified 27 October 2026)

| Priority | Source and link | Exercises and labels | Appropriate use | Licensing / limits |
| --- | --- | --- | --- | --- |
| **1** | [Mendeley: 7 gym exercises, good/bad form (v3, 10 Aug 2026)](https://data.mendeley.com/datasets/kgbb3yn47p/3) | **7 movements**, including **push-up, alternating dumbbell bicep curl, overhead dumbbell triceps extension, standing side bend**; also squat, shoulder press, bent-over row. **26 subjects**, front/side/diagonal views. Good/bad **video-level** quality labels and person IDs. | Best immediate extension for controlled good/bad multi-view gym study, not for named knee/elbow/lumbar errors without extra labels. | **CC BY 4.0**; cite DOI 10.17632/kgbb3yn47p.3. The alternate ZIP groupings represent the **same videos**; download only one organization (~2.9 GB), not 3 to avoid duplicate leakage. |
| **2** | [MM-Fit](https://mmfit.github.io/), [video archive](https://doi.org/10.5281/zenodo.7607736) | **10 exercises**, including push-up, lunge, sit-up, bicep curl, tricep extension, lateral raise, jumping jack. Includes RGB, depth, body-pose estimates, sensor data. | Multi-exercise classification / rep counting / temporal movement experiments. | Exercise labels do **not** establish quality or correctness. RGB/depth videos are very large; check archive-specific permissions/licensing before any redistribution. |
| **3** | [UI-PRMD original paper](https://doi.org/10.3390/data3010002), [Zenodo archives](https://zenodo.org/records/23104965) | **10 rehabilitation movements**, including lunge, side lunge, sit-to-stand and shoulder actions; skeletal angles/positions, correct and intentionally incorrect executions. | Pose-based quality research with cross-sensor evaluation. | **ODC PDDL** cited in original dataset paper. Vicon/Kinect skeletons do not directly match browser 33-point MediaPipe; map joints carefully and evaluate domain shift. |
| **4** | [MEx - UCI exercise dataset](https://archive.ics.uci.edu/dataset/500/mex) | **7 physiotherapy movements**, 30 subjects, depth/IMU/pressure sensors. | Exercise recognition/counting and multimodal benchmarks. | **CC BY 4.0**. Not ready-made per-error gym labels. |
| **5** | [Yoga-82](https://sites.google.com/view/yoga-82/home) | **82 yoga pose classes** and hierarchical labels; image URLs and train/test splits. | Yoga pose identification only. | **Non-commercial research/education**; individual linked images have separate copyright owners. No technique-error or repetition labels. |
| **6** | [FitAQA metadata benchmark](https://huggingface.co/datasets/Kelly0510/FitAQA) | Metadata referencing 30 bodyweight exercises and temporally localized quality QA tasks. | Error-taxonomy design and future benchmark research. | **Videos are not redistributed**; must obtain originals and obey each source's rights and split policy. Inspect benchmark use terms before training. |
| Existing | [Fitness-AQA](https://zenodo.org/records/7310289), [author GitHub](https://github.com/ParitoshParmar/Fitness-AQA) | Squat, OHP, Barbell Row original source annotations. | Existing six source error models; reference benchmark. | **Non-commercial use** in dataset terms. Do not treat repeatedly checked source validation as independent TEST evidence. |

## Recommended M20 experiment

1. Download a **single representation** of the 2026 Mendeley 7-exercise
   dataset to an ignored local directory, **not** GitHub. Verify source
   checksum, provenance, subject IDs and view labels. Confirm intended use
   and follow CC BY 4.0 attribution.
2. Focus on **Push-up and Bicep Curl** first. Build a new **exercise-specific**
   manifest and label dictionary using the dataset's actual **good/bad**
   categories. Do not rename them to detailed error types.
3. Split **by subject before extracting training windows or frames**. Keep
   a separate truly unseen participant holdout and reserve the final TEST
   group for one final report. All views of one person remain in the same
   split; the duplicated ZIP organizations are never double-counted.
4. Extract MediaPipe landmark sequences, analyze missingness by camera
   angle and subject, and measure Tasks Vision browser vs Python source
   extractor differences. Try static pose baseline, then temporal models
   for exercises requiring motion. Record balanced accuracy, macro-F1,
   per-subject generalization, precision-recall, confusion matrix,
   calibration, pose rejection rate and inference latency.
5. Deploy a **new model registry namespace per exercise, feature schema,
   label semantics and provenance**. Existing RFs must not be reused or
   silently offered for Push-up/Curl. UX should expose only models that
   have independently passed source and browser parity evaluation.
6. Preserve original media privacy, licensing, citations and train/test
   provenance. Research scores are not evidence of safety or injury risk.

The rear-facing mobile camera introduced in M19 is useful for collecting
separately consented multi-angle evaluations, but M19 does **not** save
videos or automatically record participants.

## Sources

- Mendeley Data (2026), DOI: 10.17632/kgbb3yn47p.3
- MM-Fit official site and Zenodo archive
- Vakanski et al., *Data* (2018), UI-PRMD; Zenodo download
- UCI MEx dataset DOI: 10.24432/C59K6T
- Yoga-82 research dataset terms
- FitAQA Hugging Face metadata and original provider terms
- Fitness-AQA source benchmark on Zenodo (non-commercial only)
