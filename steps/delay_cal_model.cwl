class: CommandLineTool
cwlVersion: v1.2
id: delay_cal_model
label: Delay cal model
doc: |
    Create one or more skymodels for use in the self-calibration.
    Default behaviour is to use VLASS to generate the skymodel.

baseCommand: skynet.py

inputs:
    - id: msin
      type: Directory
      doc: Input data in MeasurementSet format.
      inputBinding:
        position: 0

    - id: delay_calibrator
      type: File
      doc: Catalogue of delay calibrators.
      inputBinding:
        position: 1
        prefix: --delay-cal-file
        separate: true
    - id: model_image
      type: File?
      doc: Image to use in skynet
      inputBinding:
        position: 2
        prefix: --model-image
        separate: true
        valueFrom: $(self.basename)


    - id: process_all
      type: boolean?
      default: false
      doc: |
        If true, create a model for every entry in `delay_calibrator`.
        Otherwise, create one only for the first entry.
      inputBinding:
        position: 1
        prefix: --process-all

outputs:
    - id: skymodel
      type:
        - File[]
      outputBinding:
        glob: skymodel*.txt
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
