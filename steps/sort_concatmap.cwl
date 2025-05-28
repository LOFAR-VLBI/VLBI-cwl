class: CommandLineTool
cwlVersion: v1.2
id: sort_concatmap
label: Sort concatenate map
doc: |
    Sorts the subbands into a given number
    of regularly spaced frequency groups.

baseCommand: ./sort_times_into_freqGroups.py

inputs:
  - id: msin
    type:
      - Directory[]
    inputBinding:
      position: 1
    doc: Input MeasurementSets to be sorted.

  - id: numbands
    type: int?
    default: 10
    inputBinding:
      prefix: --numbands
      separate: true
      position: 0
    doc: The number of elements in each group.

  - id: DP3fill
    type: boolean?
    default: true
    inputBinding:
      prefix: --DP3fill
      position: 0
    doc: |
        Add dummy file names for missing frequencies,
        so that DP3 can fill the data with flagged dummy data.

  - id: stepname
    type: string?
    default: '.dp3-concat'
    inputBinding:
      prefix: --stepname
      separate: true
      position: 0
    doc: |
        A string to be appended to the file names of the output files.

  - id: mergeLastGroup
    type: boolean?
    default: False
    inputBinding:
      prefix: --mergeLastGroup
      separate: true
      position: 0
    doc: |
        Add dummy file names for missing frequencies,
        so that DP3 can fill the data with flagged dummy data.

  - id: truncateLastSBs
    type: boolean?
    default: False
    inputBinding:
      prefix: --truncateLastSBs
      separate: true
      position: 0
    doc: |
        Add dummy file names for missing frequencies,
        so that DP3 can fill the data with flagged dummy data.

  - id: firstSB
    type: int?
    default: null
    inputBinding:
      prefix: --firstSB
      separate: true
      position: 0
    doc: |
        If set, reference the grouping of
        files to this station subband.

  - id: linc_libraries
    type: File[]
    doc: |
      Scripts and reference files from the
      LOFAR INitial calibration pipeline.
      Must contain `sort_times_into_freqGroups.py`.

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.linc_libraries)

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

outputs:
  - id: filenames
    type: File
    outputBinding:
        glob: filenames.json
    doc: |
        A list of filenames that have to
        be concatenated, in JSON format.

  - id: groupnames
    type: string[]
    outputBinding:
        loadContents: true
        glob: out.json
        outputEval: $(JSON.parse(self[0].contents).groupnames)
    doc: A string of names of the frequency groups.

  - id: logfile
    type: File
    outputBinding:
      glob: sort_concatmap.log
    doc: The file containing the stdout output from the step.

stdout: sort_concatmap.log
stderr: sort_concatmap_err.log

