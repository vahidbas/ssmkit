# configure a target for tikz images
file(GLOB LATEX_TIKZ_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/image"
"${CMAKE_CURRENT_SOURCE_DIR}/image/*.tex")

# make image directory in the binary tree
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/doc/image")

find_package(LATEX)
find_package(ImageMagick COMPONENTS convert)

if (PDFLATEX_COMPILER AND ImageMagick_FOUND)
# we still need to check for tikz, how?
  foreach(_file ${LATEX_TIKZ_FILES})
    string(REPLACE ".tex" ".png" _file_o ${_file})
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/doc/image/${_file_o}
      PRE_BUILD
      COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/image/${_file}
      ${CMAKE_BINARY_DIR}/doc/image/${_file} && pdflatex -shell-escape ${_file}
      WORKING_DIRECTORY
      "${CMAKE_BINARY_DIR}/doc/image/"
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/image/${_file} 
      COMMENT "Generating ${_file_o}"
      )
    list(APPEND _tikz_images "${CMAKE_BINARY_DIR}/doc/image/${_file_o}")
  endforeach()
  #  add_custom_target(tikz_images
  #  COMMAND
  #  DEPENDS ${_images}
  #  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  #  COMMENT "Generating tikz diagrams"
  #  )
else ()
  message(STATUS "Latex ${LATEX_FOUND} and/or Imagemagick ${ImageMagick_FOUND} not found, no way to generate images")
# TODO: put a place holder instead of images..
endif ()


# find regular images
file(GLOB IMAGE_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/image"
"${CMAKE_CURRENT_SOURCE_DIR}/image/*.png"
"${CMAKE_CURRENT_SOURCE_DIR}/image/*.svg")

# message(STATUS ${IMAGE_FILES})
# copy regular images
foreach(_file ${IMAGE_FILES})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/doc/image/${_file}
      PRE_BUILD
      COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/image/${_file}
      ${CMAKE_BINARY_DIR}/doc/image/${_file}
      WORKING_DIRECTORY
      "${CMAKE_BINARY_DIR}/doc/image/"
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/image/${_file} 
      COMMENT "Copying ${_file_o}"
      )
    list(APPEND _regular_images "${CMAKE_BINARY_DIR}/doc/image/${_file}")
endforeach()
  add_custom_target(doc_images
    COMMAND
    DEPENDS ${_regular_images} ${_tikz_images}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Generating tikz diagrams and copyng images"
    )
