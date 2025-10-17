cwlVersion: v1.2
class: Workflow
id: auto_selfcal
doc: Performs direction-dependent self-calibration with full ILT calibrator data.

inputs:
  - id: msin
    type: Directory
    doc: Input MeasurementSet from calibrator source.

  - id: dd_dutch_solutions
    type: File?
    doc: Provide already obtained direction-dependent h5parm solutions for the Dutch LOFAR array to pre-apply before international LOFAR calibration.

  - id: phasediff_score_csv
    type: File?
    doc: CSV with DD selection positions and phasediff scores.

  - id: model_cache
    type: string?
    doc: Neural network cache directory.

steps:
    - id: find_closest_h5
      in:
        - id: h5parm
          source: dd_dutch_solutions
        - id: ms
          source: msin
      out:
        - closest_h5
      when: $(inputs.h5parm != null)
      run: ../../steps/find_closest_h5.cwl

    - id: addCS
      in:
        - id: ms
          source: msin
        - id: h5parm
          source:
            - find_closest_h5/closest_h5
          pickValue: the_only_non_null
      out:
        - addCS_out_h5
      when: $(inputs.h5parm != null)
      run: ../../steps/addCS.cwl

    - id: applycal
      in:
        - id: ms
          source: msin
        - id: h5parm
          source: addCS/addCS_out_h5
      out:
        - ms_out
      when: $(inputs.h5parm != null)
      run: ../../steps/applycal.cwl

    - id: make_dd_config
      in:
        - id: phasediff_output
          source: phasediff_score_csv
        - id: ms
          source: msin
        - id: dd_dutch_solutions
          source: dd_dutch_solutions
          valueFrom: $(self != null)
      out:
        - dd_config
      run: ../../steps/make_dd_config.cwl

    - id: run_facetselfcal
      in:
        - id: msin
          source:
            - applycal/ms_out
            - msin
          pickValue: first_non_null
        - id: configfile
          source: make_dd_config/dd_config
        - id: model_cache
          source: model_cache
      out:
        - h5_facetselfcal
        - selfcal_images
        - solution_inspection_images
        - fits_image
      run: ../../steps/facet_selfcal_auto.cwl

    - id: addCS_selfcal
      in:
        - id: ms
          source: msin
        - id: h5parm
          source: run_facetselfcal/h5_facetselfcal
      out:
        - addCS_out_h5
      run: ../../steps/addCS.cwl

    - id: merge_all_in_one
      in:
        - id: first_h5
          source: addCS/addCS_out_h5
        - id: second_h5
          source: addCS_selfcal/addCS_out_h5
      out:
        - merged_h5
      when: $(inputs.first_h5 != null)
      run: ../../steps/merge_in_one_dir.cwl


requirements:
  - class: InlineJavascriptRequirement

outputs:
  - id: merged_h5
    type: File
    outputSource:
      - merge_all_in_one/merged_h5
      - addCS_selfcal/addCS_out_h5
    pickValue: first_non_null

  - id: fits_images
    type: File
    outputSource: run_facetselfcal/fits_image

  - id: selfcal_inspection_images
    type: File[]
    outputSource: run_facetselfcal/selfcal_images

  - id: solution_inspection_images
    type: Directory[]
    outputSource: run_facetselfcal/solution_inspection_images