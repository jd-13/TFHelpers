function getModelNameFromFile(file) {
    return file.webkitRelativePath.split("/")[1];
}

const main = function() {

    let modelFiles;
    let models = [];

    // Activate the first button and preview when a file is selected
    $("#modelDirInput").change(function(event) {

        // Load the directory
        modelFiles = Array.from(event.currentTarget.files);

        // Each directory will contain multiple files so this array will have multiple entries.
        // We'll just filter by this file to prevent duplicates
        const filterFilename = "model.ckpt.meta";

        modelFiles.forEach(file => {
            if (file.name == filterFilename) {
                const modelTitle = getModelNameFromFile(file);
                const modelTimeStamp = file.webkitRelativePath.split("/")[2];

                // Now create a dict that has the properties of this model by parsing the directory
                // name
                let modelDict = [];
                const modelTitleSplit = modelTitle.split("-");

                modelDict["type"] = modelTitleSplit[0];
                modelDict["timestamp"] = modelTimeStamp;
                modelDict["title"] = modelTitle;

                for (let iii = 1; iii < modelTitleSplit.length - 2; iii += 2) {
                    modelDict[modelTitleSplit[iii]] = modelTitleSplit[iii + 1];
                }

                models.push(modelDict);
            }
        });

        // Now we build the checkboxes, one column at a time
        let modelColumns = Object.keys(models[0]);

        // The timestamp and title wouldn't be very useful here
        modelColumns = modelColumns.filter(element => {
            return element !== "timestamp" && element !== "title";
        });

        modelColumns.forEach(columnName => {
            // Get all the possible values for this column
            let columnValues = new Set();
            models.forEach(model => {
                columnValues.add(model[columnName]);
            });

            // Start building the html element
            const $columnHtml = $("<div class=\"col-lg-3 col-md-3 col-sm-4 col-xs-12\"></div>");
            $columnHtml.append($(`<p>${columnName}</p>`));
            const $columnFieldSet = $(`<fieldset id="${columnName}Checkboxes"></fieldset>`);

            // Add each value to the html element
            columnValues.forEach(value => {
                $columnFieldSet.append($(`<input id="${columnName}${value}" value="${value}" type="checkbox" class="filterCheckbox"/>`));
                $columnFieldSet.append($(`<label for="${columnName}${value}">${value}</label><br>`));
            });

            $columnHtml.append($columnFieldSet);
            $("#filtersRow").append($columnHtml);
        });

        // Now show the models that match the selected criteria
        $(".filterCheckbox").change(function(event) {
            const $selectedModelsRow = $("#selectedModelsRow");
            $selectedModelsRow.empty();

            // For each column, get the values that are selected
            let selectedValues = [];
            modelColumns.forEach(columnName => {
                let values = [];
                $(".filterCheckbox", $(`#${columnName}Checkboxes`)).each(function() {  
                    if (this.checked) {
                        values.push(this.value);
                    }                  
                });
                selectedValues[columnName] = values;
            });

            // For each model, check if it matches the selected values
            let matchingModels = [];
            models.forEach(model => {

                let matches = true;
                Object.keys(selectedValues).forEach(columnName => {

                    if (selectedValues[columnName].indexOf(model[columnName]) < 0) {
                        matches = false;
                    }
                });

                if (matches) {
                    matchingModels.push(model);
                    $selectedModelsRow.append(`<small>${model["title"]}&emsp;&emsp;${model["timestamp"]}</small><br>`);
                }
            });
        });
    });
}

$(document).ready(main);