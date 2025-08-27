class: Workflow
cwlVersion: v1.2
id: sort-concat-flag
label: VLBI concatenation and flagging
doc: |
    Reduces the number of MeasurementSets by concatenating
    of subbands into groups by frequency, and flags bad data
    in the resulting MeasurementSets.

inputs:
  - id: msin
    type: Directory[]
    doc: |
        Input data in MeasurementSets. A-team data
        has been removed in the setup workflow.

  - id: numbands
    type: int?
    default: 10
    doc: |
        The number of files that have to be
        grouped together in frequency.

  - id: firstSB
    type: int?
    default: null
    doc: |
        If set, reference the grouping of files
        to this station subband.

  - id: max_dp3_threads
    type: int?
    default: 5
    doc: |
        The maximum number of threads that DP3
        should use per process.

  - id: aoflagger_memory_fraction
    type: int?
    default: 15
    doc: |
        The fraction of the node's memory that
        will be used by AOFlagger (and should be
        available before an AOFlagger job can start).

  - id: rfi_strategy
    doc: The RFI strategy to use in flagging.
    type: File?
    default:
      class: File
      location: /usr/local/share/linc/rfistrategies/lofar-default.lua

steps:
  - id: get_memory
    in:
      - id: fraction
        source: aoflagger_memory_fraction
        valueFrom: $(self)
    out:
      - id: memory
    run: ../steps/get_memory_fraction.cwl
    label: Get memory fraction

  - id: sort_concatenate
    in:
      - id: msin
        source: msin
      - id: numbands
        source: numbands
      - id: firstSB
        source: firstSB
      - id: stepname
        default: '_pre-cal.ms'
    out:
      - id: filenames
      - id: groupnames
      - id: logfile
    run: ../steps/sort_concatmap.cwl
    label: sort_concatmap
  - id: concatenate-flag
    in:
      - id: msin
        source:
          - msin
      - id: group_id
        source: sort_concatenate/groupnames
      - id: groups_specification
        source: sort_concatenate/filenames
      - id: max_dp3_threads
        source: max_dp3_threads
      - id: aoflagger_memory
        source: get_memory/memory
      - id: rfi_strategy
        source: rfi_strategy
    out:
      - id: msout
      - id: concat_flag_statistics
      - id: aoflag_logfile
      - id: concatenate_logfile
    run: ./subworkflows/concatenation.cwl
    scatter: group_id
    label: concatenation-flag
  - id: concat_flags_join
    in:
      - id: flagged_fraction_dict
        source:
          - concatenate-flag/concat_flag_statistics
      - id: filter_station
        default: ''
      - id: state
        default: concat
    out:
      - id: flagged_fraction_antenna
      - id: logfile
    run: ../steps/findRefAnt_join.cwl
    label: initial_flags_join
  - id: concatenate_logfiles_concatenate
    in:
      - id: file_list
        source:
          - concatenate-flag/concatenate_logfile
      - id: file_prefix
        default: concatenate
    out:
      - id: output
    run: ../steps/concatenate_files.cwl
    label: concatenate_logfiles_concatenate
  - id: concatenate_logfiles_aoflagging
    in:
      - id: file_list
        linkMerge: merge_flattened
        source: concatenate-flag/aoflag_logfile
      - id: file_prefix
        default: AOflagging
    out:
      - id: output
    run: ../steps/concatenate_files.cwl
    label: concat_logfiles_AOflagging

  - id: summary
    in:
      - id: flagFiles
        source: concat_flags_join/flagged_fraction_antenna
      - id: run_type
        default: concatenate_flag
      - id: min_unflagged_fraction
        default: 0.5
      - id: refant
        default: CS001HBA0
    out:
      - id: summary_file
      - id: logfile
    run: ../steps/summary.cwl
    label: summary

  - id: save_logfiles
    in:
      - id: files
        linkMerge: merge_flattened
        source:
            - sort_concatenate/logfile
            - concatenate_logfiles_concatenate/output
            - concatenate_logfiles_aoflagging/output
            - concat_flags_join/logfile
            - summary/logfile
      - id: sub_directory_name
        default: 'sort-concat-flag'
    out:
      - id: dir
    run: ../steps/collectfiles.cwl
    label: save_logfiles

outputs:
    - id: logdir
      outputSource: save_logfiles/dir
      type: Directory
      doc: |
        The directory containing all the stdin
        and stderr files from the workflow.

    - id: msout
      outputSource: concatenate-flag/msout
      type: Directory[]
      doc: |
        An array of MeasurementSets
        containing the concatenated data.

    - id: concat_flags
      type: File
      outputSource: concat_flags_join/flagged_fraction_antenna
      doc: |
        A JSON formatted file containing flagging statistics
        of the MeasurementSet data after concatenation.

    - id: summary_file
      type: File
      outputSource: summary/summary_file
      doc: |
          Workflow summary statistics in JSON format.

requirements:
    - class: SubworkflowFeatureRequirement
    - class: ScatterFeatureRequirement
    - class: MultipleInputFeatureRequirement
    - class: InlineJavascriptRequirement
    - class: StepInputExpressionRequirement
