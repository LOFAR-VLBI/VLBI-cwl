cwlVersion: v1.2
class: Workflow
id: ddcal_val
doc: |
    Performing validation on direction-dependent calibration images and solutions.
    Validation is based on phase noise statistics and image inspection using peak flux,
    RMS, and neural network image validation.

inputs:
  - id: images
    type: File[]
    doc: FITS images of calibrator sources

  - id: h5parm
    type: File[]
    doc: h5parm calibration solution files

  - id: model_cache
    type: string?
    doc: Neural network cache directory

steps:
  - id: validate_images
    in:
       - id: images
         source: images
       - id: model_cache
         source: model_cache
    out:
       - validation_csv
    run: ../../steps/validate_images.cwl

  - id: validate_solutions
    in:
       - id: solutions
         source: h5parm
       - id: mode
         default: "DD"
    out:
       - validation_csv
    run: ../../steps/validate_solutions.cwl

  - id: final_validation
    in:
       - id: validation_images_csv
         source: validate_images/validation_csv
       - id: validation_solutions_csv
         source: validate_solutions/validation_csv
       - id: solutions
         source: h5parm
       - id: images
         source: images
    out:
       - output_images
       - output_solutions
       - validate_csv
    run: ../../steps/copy_dd_validated_ims_sols.cwl

outputs:

  - id: h5parm_selected
    type: File[]
    outputSource: final_validation/output_solutions
    doc: Final selected h5parm calibration solution files

  - id: images_selected
    type: File[]
    outputSource: final_validation/output_images
    doc: Final selected FITS images

  - id: validate_csv
    type: File
    outputSource: final_validation/validate_csv
    doc: Validation CSV file
