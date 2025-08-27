class: CommandLineTool
cwlVersion: v1.2
id: makeparsets
label: Make concat parsets
doc: |
    Generate direction concatenation parsets

baseCommand: concat_with_dummies

inputs:
  - id: msin
    type: Directory[]
    inputBinding:
        prefix: "--msin"
        position: 1
        separate: true
    doc: Input data in MeasurementSet format.
  - id: dysco_bitrate
    type: int?
    doc: Number of bits per float used for columns containing visibilities.
    default: 8
    inputBinding:
        prefix: "--bitrate"
        position: 2
        separate: true

outputs:
  - id: concat_parsets
    doc: |
        The output data with corrected
        data in MeasurementSet format.
    type: File[]
    outputBinding:
      glob: '*.parset'

  - id: logfile
    type: File[]
    outputBinding:
      glob: python_concat*.log
    doc: |
        The files containing the stdout
        and stderr from the step.

arguments:
  - --make_only_parset
  - --only_basename

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.msin)
        writable: false

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: python_concat.log
stderr: python_concat_err.log
