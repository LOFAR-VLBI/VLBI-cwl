class: Workflow
cwlVersion: v1.2
id: concatenation
label: Concatenation
doc: |
    Concatenation selects MeasurementSets from the
    input data based on predetermined frequency
    grouping, and concatenates the resulting groups.
    Optionally, the concatenated MeasurementSets can
    be flagged, in which case DP3 memory constraints
    and LINC flagging strategies must be given as inputs.

inputs:
  - id: msin
    type: Directory[]
    doc: |
        Input data in MeasurementSets. A-team
        data has been removed in `setup` workflow.

  - id: group_id
    type: string
    doc: The name of the final concatenated MeasurementSet.

  - id: groups_specification
    type: File
    doc: |
        A list of filenames that have to
        be concatenated, in JSON format.

  - id: max_dp3_threads
    type: int?
    default: 5
    doc: The maximum number of threads DP3 should use per process.

  - id: aoflagger_memory
    type: int?
    doc: |
        The amount of memory in mebibytes that should be available
        for an AOFlagger flagging job. Must be set if the concatenated
        data should be flagged.

  - id: strategy
    doc: The RFI strategy to use if flagging.
    type: File?

steps:
  - id: filter_ms_group
    in:
      - id: group_id
        source: group_id
      - id: groups_specification
        source: groups_specification
      - id: measurement_sets
        source: msin
    out:
      - id: selected_ms
    run: ../../steps/filter_ms_group.cwl
    label: filter_ms_group
  - id: dp3_concat
    in:
      - id: msin
        source: msin
      - id: msin_filenames
        source: filter_ms_group/selected_ms
      - id: msout_name
        source: group_id
      - id: max_dp3_threads
        source: max_dp3_threads
    out:
      - id: msout
      - id: flagged_statistics
      - id: logfile
    run: ../../steps/dp3_concat.cwl
    label: dp3_concat
  - id: AOflagging
    in:
      - id: msin
        source: dp3_concat/msout
      - id: max_dp3_threads
        source: max_dp3_threads
      - id: memory
        source: aoflagger_memory
        valueFrom: $(self)
      - id: strategy
        source: strategy
    out:
      - id: msout
      - id: logfile
    when: $((inputs.memory != null) && (inputs.strategy != null))
    run: ../../steps/aoflagger.cwl
    label: AOflagging

  - id: concat_logfiles_aoflagging
    in:
      - id: file_list
        linkMerge: merge_flattened
        source:
          - AOflagging/logfile
        pickValue: all_non_null
      - id: file_prefix
        default: AOflagging
      - id: memory
        source: aoflagger_memory
    out:
      - id: output
    when: $(inputs.memory != null)
    run: ../../steps/concatenate_files.cwl
    label: concat_logfiles_AOflagging
  - id: dp3_concatenate_logfiles
    in:
      - id: file_list
        source:
            - dp3_concat/logfile
      - id: file_prefix
        default: dp3_concatenation
    out:
      - id: output
    run: ../../steps/concatenate_files.cwl
    label: dp3_concatenate_logfiles

outputs:
  - id: msout
    outputSource:
        - AOflagging/msout
        - dp3_concat/msout
    pickValue: first_non_null
    type: Directory
    doc: The data in a concatenated MeasurementSet.

  - id: concatenate_logfile
    outputSource: dp3_concatenate_logfiles/output
    type: File
    doc: |
        The file containing the stdout and
        stderr from the dp3_concatenate step.

  - id: aoflag_logfile
    outputSource:
        - concat_logfiles_aoflagging/output
    pickValue: all_non_null
    type: File
    doc: |
        The file containing the stdout and
        stderr from the AOflagging step.

  - id: concat_flag_statistics
    type: string
    outputSource: dp3_concat/flagged_statistics
    doc: |
        A JSON formatted file containing flagging
        statistics of the data after concatenation.

requirements:
    - class: InlineJavascriptRequirement
    - class: StepInputExpressionRequirement
