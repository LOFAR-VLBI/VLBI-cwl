class: Workflow
cwlVersion: v1.2
id: delay_cal_run
doc: |-
  Selfcal processing for the delay calibrator. It calculates scalarphasediff score
  and generates improved settings for the facetselfcal run.

inputs:
  - id: msin
    type: Directory
    doc: phaseup and concatted delay calibrator MS
  - id: delay_calibrator
    type: File
    doc: delay_calibrators.csv file from plot_field.py
  - id: image_catalogue
    type: File
    doc: image_catalogue.csv file from plot_field.py
  - id: model_image
    type: File?
    doc: Input skymodel of delay calibrator
  - id: phaseup_config
    type: File
    doc: Config file defining phasediff score selfcal config.
  - id: number_cores
    type: int?
    default: 12
    doc: |
      Number of cores to use per job for tasks with
      high I/O or memory.

steps:
  - id: delay_cal_model
    label: delay_cal_model
    in:
      - id: msin
        source: msin
      - id: delay_calibrator
        source: delay_calibrator
      - id: model_image
        source: model_image
    out:
      - id: skymodel
      - id: logfile
    run: ../../steps/delay_cal_model.cwl

  - id: calc_phasediff
    in:
      - id: phasediff_ms
        source: msin
      - id: phaseup_config
        source: phaseup_config
    out:
      - id: phasediff_h5out
      - id: scalarphase_h5out
      - id: phasediff_score
    run: ../../steps/get_phasediff_delay.cwl
    label: calc_phasediff

  - id: gen_delay_config
    in:
      - id: msin
        source: msin
      - id: image_catalogue
        source: image_catalogue
      - id: model_image
        source: delay_cal_model/skymodel
        valueFrom: $(self[0])
      - id: phasediff_output
        source: calc_phasediff/phasediff_score
      - id: scalarphase_h5out
        source: calc_phasediff/scalarphase_h5out
    out:
      - id: configfile
      - id: logfile
    run: ../../steps/gen_delay_config.cwl
    label: gen_delay_config

  - id: delay_solve
    in:
      - id: msin
        source: msin
      - id: skymodel
        source: delay_cal_model/skymodel
        valueFrom: $(self[0])
      - id: configfile
        source: gen_delay_config/configfile
      - id: number_cores
        source: number_cores
    out:
      - id: h5parm
      - id: images
      - id: logfile
    run: ../../steps/facet_selfcal.cwl
    label: delay_solve


outputs:
  - id: solutions
    type: File
    outputSource: delay_solve/h5parm
    doc: |
        The calibrated solutions for the
        delay calibrator in HDF5 format.
  - id: config
    type: File
    outputSource: gen_delay_config/configfile
    doc: |
        The custom config file for the delay calibrator
  - id: pictures
    type: File[]
    outputSource: delay_solve/images
    doc: |
        The inspection plots generated
        by delay_solve.
  - id: logfile
    type: File[]
    outputSource: delay_solve/logfile

requirements:
  - class: SubworkflowFeatureRequirement
  - class: ScatterFeatureRequirement
  - class: StepInputExpressionRequirement
  - class: MultipleInputFeatureRequirement
  - class: InlineJavascriptRequirement
