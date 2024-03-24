#!/bin/bash
if [ -z "$(pip list | grep pyflakes)" ]; then
    echo
    echo "Package pyflakes is necessary but not installed."
    echo
    read -p "Do you want to install it? (y/n) " CONTINUE
    echo
    if [ "$CONTINUE" != "y" ]; then
        echo "Can not work without this package. Exiting..."
        echo
        exit 0
    fi
    pip install pyflakes
fi

AIDER_CMD="/Users/cgint/dev/aider-runtime/aider.sh"
CHECK_FILES_AND_DIRS="*.py lib/ tests/"
if [ -n "$1" ]; then
    CHECK_FILES_AND_DIRS="$1"
fi
FOUND_FLAKES=$(python -m pyflakes $CHECK_FILES_AND_DIRS)
if [ -n "$FOUND_FLAKES" ]; then
    echo
    echo "pyflakes found the following errors:"
    echo
    echo "$FOUND_FLAKES"

    echo
    read -p "Do you want to fix them using aider? (y/n) " CONTINUE
    if [ "$CONTINUE" == "y" ]; then
        rm flakes.txt
        echo "Correct the following issues found by pyflakes:" >> flakes.txt
        echo "'''" >> flakes.txt
        echo "$FOUND_FLAKES" >> flakes.txt
        echo "'''" >> flakes.txt

        echo "This is the flakes.txt file:"
        echo "-------------------------------"
        cat flakes.txt
        echo "-------------------------------"

        read -p "Do you want to run aider with auto-correct, save, commit to git? (y/n) " CONTINUE
        if [ "$CONTINUE" == "y" ]; then
            $AIDER_CMD --yes --message-file flakes.txt
        fi
        rm flakes.txt

    fi
else
    echo
    echo "No flakes found. Have a nice day!"
    echo
fi
