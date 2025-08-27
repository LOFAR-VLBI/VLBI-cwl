class: Workflow
cwlVersion: v1.2
id: facet_subtract
label: Facet subtraction
doc: Use WSClean to predict and subtract model data, to split all facets into separate MeasurementSets.

inputs:
    - id: msin
      type: Directory[]
      doc: Unaveraged MeasurementSets with coverage of the target directions.
    - id: model_image_folder
      type: Directory
      doc: Folder with 1.2" model images.
    - id: h5parm
      type: File
      doc: Merged h5parms
    - id: facetselfcal
      type: Directory
      doc: facetselfcal directory.
    - id: copy_to_local_scratch
      type: boolean?
      doc: Specific option for using --bypass-file-store on the Spider cluster to run predict and subtract on local scratch.
      default: false
    - id: ncpu
      type: int?
      doc: Number of cores to use during predict and subtract.
      default: 16
    - id: dysco_bitrate
      type: int?
      doc: Number of bits per float used for columns containing visibilities.
      default: 8

steps:
    - id: get_facet_layout
      label: Get DS9 facet layout
      in:
        - id: msin
          source: msin
          valueFrom: $(self[0])
        - id: h5parm
          source: h5parm
        - id: facetselfcal
          source: facetselfcal
      out:
        - id: facet_regions
      run: ../steps/get_facet_layout.cwl

    - id: get_model_images
      label: Get WSClean model images
      in:
        - id: model_image_folder
          source: model_image_folder
      out:
        - id: filtered_model_image_folder
      run: ../steps/gather_model_images.cwl

    - id: subtract_fov_wsclean
      label: Subtract complete FoV
      in:
         - id: msin
           source: msin
         - id: h5parm
           source: h5parm
         - id: facet_regions
           source: get_facet_layout/facet_regions
         - id: model_image_folder
           source: get_model_images/filtered_model_image_folder
         - id: copy_to_local_scratch
           source: copy_to_local_scratch
         - id: ncpu
           source: ncpu
      out:
         - subtracted_ms
      run: ../steps/subtract_fov_wsclean.cwl
      scatter: msin

    - id: split_polygons
      label: Split polygon file
      in:
         - id: facet_regions
           source: get_facet_layout/facet_regions
         - id: h5parm
           source: h5parm
      out:
         - id: polygon_info
         - id: polygon_regions
      run: ../steps/split_polygons.cwl

    - id: predict_facet
      label: Predict a polygon back in empty MS
      in:
         - id: subtracted_ms
           source: subtract_fov_wsclean/subtracted_ms
         - id: polygon_region
           source: split_polygons/polygon_regions
         - id: h5parm
           source: h5parm
         - id: polygon_info
           source: split_polygons/polygon_info
         - id: model_image_folder
           source: get_model_images/filtered_model_image_folder
         - id: copy_to_local_scratch
           source: copy_to_local_scratch
         - id: ncpu
           source: ncpu
      out:
         - facet_ms
      run: ../steps/predict_facet.cwl
      scatter: [subtracted_ms, polygon_region]
      scatterMethod: flat_crossproduct

    - id: make_concat_parset
      label: Make concat parsets
      in:
         - id: msin
           source: predict_facet/facet_ms
         - id: dysco_bitrate
           source: dysco_bitrate
      out:
         - id: concat_parsets
      run: ../steps/make_concat_parsets.cwl

    - id: concat_facets
      label: Run DP3 parsets
      in:
        - id: parset
          source: make_concat_parset/concat_parsets
        - id: msin
          source: predict_facet/facet_ms
      out:
        - id: msout
      run: ../steps/dp3_parset.cwl
      scatter: parset

requirements:
  - class: MultipleInputFeatureRequirement
  - class: ScatterFeatureRequirement

outputs:
    - id: facet_ms
      type: Directory[]
      outputSource:
        - concat_facets/msout
        - predict_facet/facet_ms
      pickValue: first_non_null
    - id: polygon_info
      type: File
      outputSource: split_polygons/polygon_info
    - id: polygon_regions
      type: File[]
      outputSource: split_polygons/polygon_regions
