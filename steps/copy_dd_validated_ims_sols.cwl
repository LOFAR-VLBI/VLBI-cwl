cwlVersion: v1.2
class: CommandLineTool
id: validation
doc: Filter FITS images and calibration solutions according to validation scores

baseCommand: copy_dd_validated_ims_sols.py

inputs:
    - id: images
      type: File[]
      doc: FITS images
      inputBinding:
        position: 1
        prefix: "--images"
        separate: true
    - id: solutions
      type: File[]
      doc: Calibration solutions
      inputBinding:
        position: 2
        prefix: "--h5parms"
        separate: true
    - id: validation_images_csv
      type: File
      doc: CSV with image validation information
      inputBinding:
        position: 3
        prefix: "--validation_images_csv"
        separate: true
    - id: validation_solutions_csv
      type: File
      doc: CSV with calibration solutions validation information
      inputBinding:
        position: 4
        prefix: "--validation_solutions_csv"
        separate: true

arguments:
  - --copy_selected
  - --error_on_bad_solutions

outputs:
    - id: output_images
      type: File[]
      doc: Selected FITS images
      outputBinding:
        glob: "select_*.fits"
    - id: output_solutions
      type: File[]
      doc: Selected calibration solutions
      outputBinding:
        glob: "select_*.h5"
    - id: validate_csv
      type: File
      doc: Validation CSV file
      outputBinding:
        glob: validate.csv
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step
      outputBinding:
        glob: validate*.log

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: validate.log
stderr: validate_err.log