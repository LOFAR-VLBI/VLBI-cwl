class: Workflow
cwlVersion: v1.2
id: image_intermediate_resolution
label: Intermediate resolution imaging
doc: |
    This workflow will make an intermediate resolution image.

requirements:
    - class: InlineJavascriptRequirement
    - class: StepInputExpressionRequirement
    - class: ScatterFeatureRequirement

inputs:
    - id: msin
      type: Directory[]
      doc: MeasurementSets that will be imaged.

    - id: number_cores
      type: int?
      default: 12
      doc: The minimum number of cores that should be available for steps that require high I/O.

    - id: dd_solutions
      type: File
      doc: Direction-dependent calibration solutions in the form of a multi-direction H5parm.
      
    - id: image_size
      type: int[]?
      default: [22500, 22500]
      doc: Size of the image to make in pixels.

    - id: pixel_scale
      type: float?
      default: 0.4
      doc: Pixel size in arcseconds.

steps:
    - id: average_data
      label: average
      in:
        - id: msin
          source: msin
      out:
        - id: ms_avg
      run: ../steps/average_intermediate_resolution.cwl

    - id: make_facet_layout
      label: facet_layout
      in:
        - id: msin
          source: average_data/ms_avg
        - id: h5parm
          source: dd_solutions
        - id: imsize
          source: image_size
          valueFrom: $(inputs.image_size[0])
        - id: pixelscale
          source: pixel_scale
        - id: regionfile
          valueFrom: "facets_1p5asec.reg"
      out:
        - id: facet_regions
      run: ../steps/get_facet_layout.cwl

    - id: make_intermediate_resolution_image
      label: intermediate_resolution_image
      in:
        - id: msin
          source: msin
        - id: cores
          source: number_cores
        - id: size
          source: image_size
        - id: scale
          source: pixel_scale
          valueFrom: $(self.toString() + "asec")
        - id: taper-gaussian
          valueFrom: $(1.5.toString() + "asec")
        - id: facet-regions
          source: make_facet_layout/facet_regions
        - id: dd_solutions
          source: dd_solutions
        - id: facet-options
          valueFrom: |
            $({
              "facet-solutions": inputs.dd_solutions,
              "soltabs": ["amplitude000", "phase000"]
            })
      out:
        - id: MFS_images
        - id: channel_model_images

      run: ../steps/wsclean.cwl

outputs:
    - id: facet_region
      outputSource:
        - make_facet_layout/facet_regions
      type: File
      doc: |
        DS9 region file containing the facet layout.
    - id: MFS_images
      outputSource:
        - make_intermediate_resolution_image/MFS_images
      type: File[]
      doc: |
        Final MFS FITS images at intermediate resolution.
    - id: channel_model_images
      outputSource:
      - make_intermediate_resolution_image/channel_model_images
      type: File[]
      doc: |
        Final channel model FITS images of the intermediate resolution image.
