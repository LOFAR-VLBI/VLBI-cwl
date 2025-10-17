class: CommandLineTool
cwlVersion: v1.2
id: gen_delay_config
label: Geneate DI configfile
doc: |
    Script to generate the appropriate config file settings

baseCommand: make_config_dical.py

inputs:
    - id: msin
      type: Directory
      doc: Input data in MeasurementSet format.
      inputBinding:
        position: 0
        prefix: --ms
        separate: true

    - id: image_catalogue
      type: File?
      doc: Catalogue of imaging targets.
      inputBinding:
        position: 1
        prefix: --imagecat
        separate: true
    - id: model_image
      type: File
      doc: Model image for delay calibrator.
      inputBinding:
        position: 2
        prefix: --inputmodel
        separate: true
    - id: phasediff_output
      type: File
      doc: Phasediff scores and wraps.
      inputBinding:
        position: 2
        prefix: --phasediff_output
        separate: true
    - id: scalarphase_h5out
      type: File
      doc: h5 parm containing scalar phase solutions
      inputBinding:
        position: 3
        prefix: --scalarphase-h5
        separate: true

outputs:
    - id: configfile
      type: File
      outputBinding:
        glob: '*.config.txt'
      doc: The generate facetselfcal config file.

    - id: logfile
      type: File[]
      outputBinding:
        glob: delay_cal_config*.log
      doc: |
        The files containing the stdout
        and stderr from the step.

hints:
    - class: DockerRequirement
      dockerPull: vlbi-cwl
    - class: InitialWorkDirRequirement
      listing:
        - entry: $(inputs.msin)
          writable: true

stdout: gen_delay_config.log
stderr: gen_delay_config_err.log
