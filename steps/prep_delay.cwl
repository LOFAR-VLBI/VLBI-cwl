class: CommandLineTool
cwlVersion: v1.2
id: prep_delay
label: Prepare delay
doc: |
    Converts the delay calibrator information into strings.

baseCommand: TargetListToCoords.py

inputs:
    - id: delay_calibrator
      type: File
      doc: |
        The file containing the properties and
        coordinates of the delay calibrator.
      inputBinding:
        prefix: --target_file
        separate: true

    - id: mode
      type:
        type: enum
        symbols:
          - "delay_calibration"
          - "split_directions"
      doc: |
        The name of the processing mode.
        Must be either 'delay_calibration' or 'split_directions'.
      inputBinding:
        prefix: --mode
        separate: true

requirements:
  - class: InlineJavascriptRequirement

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

outputs:
    - id: source_id
      type: string
      doc: Catalogue source ID.
      outputBinding:
        loadContents: true
        glob: out.json
        outputEval: $(JSON.parse(self[0].contents).name)

    - id: coordinates
      type: string
      doc: Catalogue source coordinates.
      outputBinding:
        loadContents: true
        glob: out.json
        outputEval: $(JSON.parse(self[0].contents).coords)

    - id: logfile
      type: File[]
      outputBinding:
        glob: prep_delay*.log
      doc: |
        The files containing the stdout
        and stderr outputs from the step.

stdout: prep_delay.log
stderr: prep_delay_err.log
