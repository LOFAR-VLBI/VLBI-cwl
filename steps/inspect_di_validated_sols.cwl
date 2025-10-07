cwlVersion: v1.2
class: CommandLineTool
id: di_validation
doc: Inspect direction-independent calibration solution validation

baseCommand: inspect_di_validated_sols.py

inputs:
    - id: solutions
      type: File[]
      doc: Calibration solutions
      inputBinding:
        position: 1
        prefix: "--h5parms"
        separate: true
    - id: validation_solutions_csv
      type: File
      doc: CSV with calibration solutions validation information
      inputBinding:
        position: 2
        prefix: "--validation_solutions_csv"
        separate: true

arguments:
  - --return_error

outputs:
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step
      outputBinding:
        glob: validate_di*.log

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: validate_di.log
stderr: validate_di_err.log