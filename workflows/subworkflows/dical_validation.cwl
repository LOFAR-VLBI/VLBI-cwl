cwlVersion: v1.2
class: Workflow
id: dical_val
doc: |
    Performing validation on direction-independent calibration solutions.
    Validation is based on phase and amplitude noise statistics.

inputs:
  - id: h5parm
    type: File[]
    doc: h5parm calibration solution files

steps:

  - id: validate_solutions
    in:
       - id: solutions
         source: h5parm
       - id: mode
         default: "DI"
    out:
       - validation_csv
    run: ../../steps/validate_solutions.cwl

  - id: final_validation
    in:
       - id: validation_solutions_csv
         source: validate_solutions/validation_csv
       - id: solutions
         source: h5parm
    out:
       - logfile
    run: ../../steps/inspect_di_validated_sols.cwl

outputs:

  - id: validate_csv
    type: File
    outputSource: validate_solutions/validation_csv
    doc: Validation CSV file
