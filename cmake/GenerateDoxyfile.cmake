# Modifies and copies Doxyfile into build directory

# Read the Doxyfile into a variable.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile" DOXYFILE_CONTENTS)

# Modify all the input path
string(REPLACE
    "./src/ssmpack"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ssmpack"
    DOXYFILE_AUXVAR "${DOXYFILE_CONTENTS}"
    )
# Modify documentation global path
string(REPLACE
    "./doc/global"
    "${CMAKE_CURRENT_SOURCE_DIR}/doc/global"
    DOXYFILE_AUXVAR "${DOXYFILE_AUXVAR}"
    )

# Change the STRIP_FROM_PATH so that it works right even in the build directory;
# otherwise, every file will have the full path in it.
string(REGEX REPLACE
    "(STRIP_FROM_PATH[ ]*=) ./"
    "\\1 ${CMAKE_CURRENT_SOURCE_DIR}/"
    DOXYFILE_AUXVAR ${DOXYFILE_AUXVAR})

# Save the Doxyfile to its new location.
file(WRITE "${DESTDIR}/Doxyfile" "${DOXYFILE_AUXVAR}")
