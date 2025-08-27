class: Workflow
cwlVersion: v1.2
id: process_ddf
doc: |-
  Uses the results from the DDF pipeline to the data to effect the following:

  * Correct the data from the Dutch stations for direction-dependent effects.
  * [Optionally] subtract a LoTSS model from the data.

  Subtracting the LoTSS skymodel prepares the data for widefield imaging by subtracting sources outside a given region,
  defaulting to the approximate FWHM of the international stations.

inputs:
  - id: msin
    type: Directory[]
    doc: Input data from which the LoTSS skymodel will be subtracted.
  - id: solsdir
    type: Directory
    doc: Path to the SOLSDIR directory of the DDF-pipeline run.
  - id: ddf_rundir
    type: Directory
    doc: Directory containing the output from DDF-pipeline.
  - id: box_size
    type: float?
    doc: |-
      Side length of a square box in degrees. The LoTSS skymodel is subtracted outside of this box.
      Defaults to 2.5 degrees.
    default: 2.5
  - id: freqavg
    type: int?
    doc: Number of frequency channels to average after the subtract has been performed. Defaults to 1 (no averaging).
    default: 1
  - id: timeavg
    type: int?
    doc: Number of time slots to average after the subtract has been performed. Defaults to 1 (no averaging).
    default: 1
  - id: ncpu
    type: int?
    doc: Number of cores to use during the subtract. Defaults to 24.
    default: 24
  - id: chunkhours
    type: float?
    doc: The range of time to predict the model for at once. Lowering this value reduces memory footprint, but can increase runtime.
  - id: do_subtraction
    type: boolean?
    default: false
    doc: When set to true, the LoTSS model will be subtracted from the DDF corrected data.

outputs:
  - id: regionbox
    type: File?
    outputSource:
      - subtract_lotss/regionbox
    pickValue: all_non_null
    doc: DS9 region file outside of which the LoTSS skymodel has been subtracted.
  - id: mslist
    type: File[]?
    outputSource:
      - subtract_lotss/mslist
    pickValue: all_non_null
    doc: Text file containing the name of the input MS from which the LoTSS skymodel hase been subtracted.
  - id: msout
    type: Directory[]
    outputSource:
      - subtract_lotss/msout
      - dp3_applycal_ddf/output_data
    pickValue: first_non_null
    doc: MSs from which the LoTSS skymodel has been subtracted.

steps:
  - id: convert_ddf_dis2
    in:
      - id: msin
        source: msin
        valueFrom: $(self[0])
      - id: ddf_solsdir
        source: solsdir
        valueFrom: $(self)
    out:
      - id: dis2_h5parm
    run: ../steps/gatherdis2.cwl

  - id: dp3_applycal_ddf
    in:
      - id: msin
        source: msin
      - id: ddf_solset
        source: convert_ddf_dis2/dis2_h5parm
    out:
      - id: output_data
      - id: logfile
    run: ../steps/dp3_applycal_ddf.cwl
    label: dp3_applycal_ddf
    scatter: msin

  - id: subtract_lotss
    in:
      - id: msin
        source: dp3_applycal_ddf/output_data
      - id: solsdir
        source: solsdir
      - id: ddf_rundir
        source: ddf_rundir
      - id: box_size
        source: box_size
      - id: freqavg
        source: freqavg
      - id: timeavg
        source: timeavg
      - id: ncpu
        source: ncpu
      - id: chunkhours
        source: chunkhours
      - id: do_subtraction
        source: do_subtraction
    out:
      - id: regionbox
      - id: mslist
      - id: msout
    label: subtract_lotss
    when: $(inputs.do_subtraction)
    run: ./subworkflows/subtract_lotss.cwl

requirements:
  - class: ScatterFeatureRequirement
  - class: StepInputExpressionRequirement
  - class: SubworkflowFeatureRequirement
  - class: InlineJavascriptRequirement

