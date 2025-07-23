class: CommandLineTool
cwlVersion: v1.2
id: select_best_directions
label: Select best directions
doc: This step uses the phasediff scores to select the best input directions by adding a suffix *_best.ms

baseCommand:
  - direction_selection.py
  - --best_score=2.3

inputs:
    - id: msin
      type: Directory[]
      doc: All input MS directions
      inputBinding:
        prefix: "--ms"
        position: 1
        separate: true
    - id: phasediff_csv
      type: File
      doc: CSV with phasediff source selection scores
      inputBinding:
        prefix: "--csv"
        position: 2
    - id: phasediff_score
      type: float
      default: 2.3
      doc: Phasediff score, see Section 3.3.1 from de Jong et al. (2024; https://arxiv.org/abs/2407.13247)
      inputBinding:
        prefix: "--best_score"
        position: 3

outputs:
    - id: best_ms
      type: Directory[]
      doc: Best directions
      outputBinding:
        glob: "*_best.ms"
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step
      outputBinding:
        glob: dir_selection*.log

requirements:
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.msin)
        writable: true

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: dir_selection.log
stderr: dir_selection_err.log
