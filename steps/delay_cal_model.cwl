class: CommandLineTool
cwlVersion: v1.2
id: delay_cal_model
label: Delay cal model
doc: |
    Creates a skymodel for use in the self-calibration.

baseCommand: skynet.py

inputs:
    - id: msin
      type: Directory
      doc: Input data in MeasurementSet format.
      inputBinding:
        position: 0

    - id: delay_calibrator
      type: File
      doc: Coordinates of a suitable delay calibrator.
      inputBinding:
        position: 1
        prefix: --delay-cal-file
        separate: true

    - id: model_image
      type: File?
      doc: Image to generate an initial delay calibrator model from.
      inputBinding:
        position: 1
        prefix: --model-image
        separate: true

outputs:
    - id: skymodel
      type: File
      outputBinding:
        glob: skymodel.txt
      doc: The skymodel of the delay calibrator.

    - id: logfile
      type: File[]
      outputBinding:
        glob: delay_cal_model*.log
      doc: |
        The files containing the stdout
        and stderr from the step.

hints:
    - class: DockerRequirement
      dockerPull: vlbi-cwl

stdout: delay_cal_model.log
stderr: delay_cal_model_err.log
