cwlVersion: v1.2
class: CommandLineTool
id: select_bright_sources
label: Select sources above flux density threshold
doc: |
      Select bright sources for which peak flux exceeds the threshold and within 2.5×2.5 deg^2 box centred on phase centre
      (see Sweijen et al. 2022 (https://arxiv.org/pdf/2202.01608); Section 3.1 from de Jong et al. 2024 (https://arxiv.org/pdf/2407.13247))


baseCommand: pre_flux_selection.py

inputs:
    - id: msin
      type: Directory[]
      doc: MeasurementSets for getting phase centre of observations.
      inputBinding:
        position: 1
        prefix: "--ms"
        valueFrom: "$(inputs.msin[0])"
        separate: true
    - id: image_cat
      type: File
      doc: A LoTSS 6" source catalogue containing the relevant imaging sources.
      inputBinding:
        position: 2
        prefix: "--catalogue"
        separate: true
    - id: peak_flux_cut
      type: float
      doc: Peak flux (Jy/beam) cut to pre-select sources from catalogue.
      inputBinding:
        prefix: "--fluxcut"
        position: 3
        separate: true

outputs:
    - id: bright_cat
      type: File
      doc: Output catalogue after selection in CSV format
      outputBinding:
        glob: "bright_cat.csv"
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step
      outputBinding:
        glob: make_dd_config*.log

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: select_bright_sources.log
stderr: select_bright_sources_err.log