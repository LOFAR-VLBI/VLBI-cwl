class: CommandLineTool
cwlVersion: v1.2
id: get_source_scores
label: Source Scores
doc: |
       This step calculates the scores from h5parm solution files to determine if a direction is suitable
       for long-baseline calibration for wide-field imaging (See Section 3.3.1 https://arxiv.org/pdf/2407.13247)

baseCommand:
  - python3

inputs:
    - id: phasediff_h5
      type: File[]
      doc: H5parm solutions from facetselfcal.
      inputBinding:
        prefix: "--h5"
        position: 1
        itemSeparator: " "
        separate: true
    - id: selfcal
      type: Directory
      doc: Path to facetselfcal directory.

outputs:
    - id: phasediff_score_csv
      type: File
      doc: csv with phase difference scores
      outputBinding:
        glob: "*.csv"
    - id: logfile
      type: File[]
      doc: log files corresponding to this step
      outputBinding:
        glob: phasediff*.log

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.phasediff_h5)
        writable: true
      - entry: $(inputs.selfcal)

arguments:
  - $( inputs.selfcal.path + '/submods/source_selection/phasediff_output.py' )

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: phasediff.log
stderr: phasediff_err.log
