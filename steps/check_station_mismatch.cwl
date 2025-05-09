class: CommandLineTool
cwlVersion: v1.2
id: check_station_mismatch
label: Check station mismatch
doc: |
    Compares the lists of stations contained in MeasurementSets
    against the list of station in the solution file and ensures
    both are consistent.

baseCommand: compareStationListVLBI.py

inputs:
    - id: msin
      type: Directory[]
      doc: Input MeasurementSets.
      inputBinding:
        position: 1

    - id: solset
      type: File
      doc: The solution set from the LINC pipeline.
      inputBinding:
        position: 0
        prefix: --solset
        separate: true

    - id: solset_name
      type: string?
      doc: Name of the solution set.
      default: vlbi
      inputBinding:
        position: 0
        prefix: --solset_name
        separate: true

    - id: filter_baselines
      type: string?
      default: "*&"
      doc: Filter constrains for the dp3_prep_target step.
      inputBinding:
        position: 0
        prefix: --filter_baselines
        separate: true

outputs:
    - id: filter_out
      type: string
      outputBinding:
        loadContents: true
        glob: out.json
        outputEval: $(JSON.parse(self[0].contents).filter_out)
      doc: |
        A JSON formatted filter command containing
        the station names to filter.

    - id: logfile
      type: File[]
      outputBinding:
        glob: compareStationMismatch*.log
      doc: |
        The files containing the stdout
        and stderr from the step.

requirements:
    - class: InlineJavascriptRequirement

hints:
    - class: DockerRequirement
      dockerPull: vlbi-cwl

stdout: compareStationMismatch.log
stderr: compareStationMismatch_err.log
