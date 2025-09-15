cwlVersion: v1.2
class: CommandLineTool
id: validate_solutions
doc: Validate calibration solution quality

baseCommand: validate_lofar_solutions.py

inputs:
    - id: solutions
      type: File[]
      doc: Calibration solutions
      inputBinding:
        position: 2
        separate: true
    - id: mode
      type:
        - type: enum
          name: Mode
          symbols: [DI, DD]
        - string
      doc: Validation mode (DI or DD)
      inputBinding:
        prefix: --mode
        separate: true
        position: 1

outputs:
    - id: validation_csv
      type: File
      doc: CSV with calibration solution validation information
      outputBinding:
        glob: "validation_solutions.csv"
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step
      outputBinding:
        glob: validate_solutions*.log

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: validate_solutions.log
stderr: validate_solutions_err.log