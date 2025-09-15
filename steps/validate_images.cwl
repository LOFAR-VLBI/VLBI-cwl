cwlVersion: v1.2
class: CommandLineTool
id: validate_images
doc: Validate image quality according to neural network scores

baseCommand: validate_lofar_images.py

inputs:
    - id: images
      type: File[]
      doc: FITS images
      inputBinding:
        position: 2
        separate: true
    - id: model_cache
      type: string?
      doc: Cache directory with Neural Network model
      inputBinding:
        position: 1
        prefix: "--nn_model_cache"
        separate: true

outputs:
    - id: validation_csv
      type: File
      doc: CSV with image validation information
      outputBinding:
        glob: "validation_images.csv"
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step
      outputBinding:
        glob: validate_images*.log

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: validate_images.log
stderr: validate_images_err.log