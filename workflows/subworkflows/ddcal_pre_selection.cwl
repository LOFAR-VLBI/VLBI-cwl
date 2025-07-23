class: Workflow
cwlVersion: v1.2
id: ddcal_pre_selection
label: DD direction selection
doc: |
   This workflow does the following:
        * DP3 prep to average measurement to the same freq/time resolution
        * Get h5parm solutions with scalarphasediff corrections from facetselfcal
        * Get solution scores using the circular standard deviation
        * Select MS with scores below 2.3
   This selection metric is described in Section 3.3.1 from de Jong et al. (2024; https://arxiv.org/pdf/2407.13247)

requirements:
  - class: ScatterFeatureRequirement
  - class: SubworkflowFeatureRequirement

inputs:
    - id: msin
      type: Directory[]
      doc: The input concatenated MS.
    - id: phasediff_score
      type: float
      default: 2.3
      doc: Phasediff-score for calibrator selection <2.3 good for DD-calibrators and <0.7 good for DI-calibrators.
    - id: lofar_helpers
      type: Directory
      doc: Path to lofar_helpers directory.
    - id: selfcal
      type: Directory
      doc: Path to selfcal directory.

steps:
    - id: Phasediff
      in:
        - id: lofar_helpers
          source: lofar_helpers
        - id: selfcal
          source: selfcal
        - id: msin
          source: msin
      out:
        - phasediff_h5out
      scatter: msin
      run:
        # start of Phasediff
        cwlVersion: v1.2
        class: Workflow
        inputs:
          - id: msin
            type: Directory
          - id: selfcal
            type: Directory
          - id: lofar_helpers
            type: Directory
        outputs:
          - id: phasediff_h5out
            type: File
            outputSource: get_phasediff/phasediff_h5out

        steps:
          - id: dp3_prep_phasediff
            label: Pre-averaging with DP3
            in:
              - id: msin
                source: msin
            out:
              - phasediff_ms
            run: ../../steps/dp3_prep_phasediff.cwl

          - id: get_phasediff
            label: Get phase difference with facetselfcal
            in:
              - id: phasediff_ms
                source: dp3_prep_phasediff/phasediff_ms
              - id: lofar_helpers
                source: lofar_helpers
              - id: selfcal
                source: selfcal
            out:
              - phasediff_h5out
            run: ../../steps/get_phasediff.cwl
        # end of Phasediff

    - id: get_selection_scores
      label: Calculate phase difference score
      in:
        - id: phasediff_h5
          source: Phasediff/phasediff_h5out
        - id: selfcal
          source: selfcal
      out:
        - phasediff_score_csv
      run: ../../steps/get_selection_scores.cwl

    - id: select_best_directions
      label: Select best directions
      in:
        - id: phasediff_csv
          source: get_selection_scores/phasediff_score_csv
        - id: msin
          source: msin
        - id: phasediff_score
          source: phasediff_score
      out:
        - best_ms
      run: ../../steps/select_best_directions.cwl

outputs:
    - id: phasediff_score_csv
      type: File
      outputSource: get_selection_scores/phasediff_score_csv
      doc: csv with scores
    - id: best_ms
      type: Directory[]
      outputSource: select_best_directions/best_ms
      doc: Final MS selection
