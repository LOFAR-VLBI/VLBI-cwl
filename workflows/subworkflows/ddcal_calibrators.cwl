cwlVersion: v1.2
class: Workflow
id: ddcal_int
doc: Performing direction-dependent self-calibration for international LOFAR stations for multiple directions.

inputs:
  - id: msin
    type: Directory[]
    doc: Input MeasurementSets from individual calibrator directions.

  - id: dd_dutch_solutions
    type: File?
    doc: Multi-directional h5parm with Dutch DD solutions.

  - id: phasediff_score_csv
    type: File?
    doc: CSV with DD selection positions and phasediff scores.

  - id: lofar_helpers
    type: Directory
    doc: LOFAR helpers directory

  - id: facetselfcal
    type: Directory
    doc: facetselfcal directory

steps:
    - id: ddcal
      in:
        - id: msin
          source: msin
        - id: dd_dutch_solutions
          source: dd_dutch_solutions
        - id: phasediff_score_csv
          source: phasediff_score_csv
        - id: lofar_helpers
          source: lofar_helpers
        - id: facetselfcal
          source: facetselfcal
      out:
        - merged_h5
        - fits_images
        - selfcal_inspection_images
        - solution_inspection_images
      run: ./auto_selfcal.cwl
      scatter: msin

    - id: multidir_merge
      in:
        - id: h5parms
          source: ddcal/merged_h5
        - id: facetselfcal
          source: facetselfcal
      out:
        - multidir_h5
      run: ../../steps/multidir_merger.cwl

    - id: flatten_images
      in:
        - id: nestedarray
          source: ddcal/selfcal_inspection_images
      out:
        - flattenedarray
      run: ../../steps/flatten.cwl

    - id: flatten_solutions
      in:
        - id: nestedarray
          source: ddcal/solution_inspection_images
      out:
        - flattenedarray
      run: ../../steps/flatten.cwl

requirements:
  - class: ScatterFeatureRequirement

outputs:
  - id: final_merged_h5
    type: File
    outputSource: multidir_merge/multidir_h5
    doc: Final merged h5parm with calibration solutions from multiple calibrator directions

  - id: selfcal_images
    type: File[]
    outputSource: ddcal/fits_images
    doc: Self-calibration images in FITS format

  - id: selfcal_inspection_images
    type: File[]
    outputSource: flatten_images/flattenedarray
    doc: Self-calibration inspection images in PNG format

  - id: solution_inspection_images
    type: Directory[]
    outputSource: flatten_solutions/flattenedarray
    doc: LoSoTo solution inspection images
