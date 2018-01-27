function getModelNameFromFile(file) {
    return file.webkitRelativePath.split("/")[1];
}

function buildModelDictFromFile(file) {
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

    return modelDict;
}

function buildTensorboardCommand(matchingModels) {
    // TODO: remove duplicate logdirs
    // Build the tensorboard command
    const $tensorboardRow = $("#tensorboardCommandRow");
    $tensorboardRow.empty();
    $tensorboardRow.append("<h3>Tensorboard command:</h3>");
    $tensorboardRow.append("<a class=\"btn btn-info\" href=\"localhost:6006\" target=\"_blank\">Tensorboard (6006)</a>");

    const $commandCopyBtn = $("<button class=\"btn btn-info\">Copy to Clipboard</button>");
    $commandCopyBtn.click(() => {
        document.getElementById("tensorboardCommand").select();
        document.execCommand("copy");
    });
    $tensorboardRow.append($commandCopyBtn);

    let logdirString = "";
    matchingModels.forEach(model => {
        logdirString += "models/" + model["title"] + ",";
    });
    logdirString = logdirString.slice(0, -1);

    $tensorboardRow.append(`<br><input id="tensorboardCommand" readonly="readonly" class="commandInput" value="tensorboard --logdir=${logdirString}">`);
}

let filter = {
    modelColumns: undefined,
    models: undefined,

    onFilterUpdate: function() {
        const $selectedModelsRow = $("#selectedModelsRow");
        $selectedModelsRow.empty();
        $selectedModelsRow.append("<h3>Matching models:</h3>");
    
        // For each column, get the values that are selected
        let selectedValues = [];
        filter.modelColumns.forEach(columnName => {
            let values = [];
            $(".filterCheckbox", $(`#${columnName}Checkboxes`)).each(function() {  
                if (this.checked) {
                    values.push(this.value);
                }                  
            });
            selectedValues[columnName] = values;
        });
    
        const matchingModels = this.getMatchingModels(filter.models, selectedValues, $selectedModelsRow);         
    
        buildTensorboardCommand(matchingModels);
    },

    getMatchingModels: function(models, selectedValues, $selectedModelsRow) {
        // TODO: put models with mulitple runs on only one row
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

        return matchingModels;
    },

    buildCheckboxes: function($filtersRow) {
        this.modelColumns.forEach(columnName => {
            // Get all the possible values for this column
            let columnValues = new Set();
            this.models.forEach(model => {
                columnValues.add(model[columnName]);
            });
    
            // Start building the html element
            const $columnHtml = $("<div class=\"col-lg-3 col-md-3 col-sm-4 col-xs-12\"></div>");
    
            const columnBtnID = `${columnName}btn`;
            $columnHtml.append($(`<button id="${columnBtnID}" class="btn btn-info">${columnName}</button>`));
    
            const columnFieldSetID = `${columnName}Checkboxes`;
            const $columnFieldSet = $(`<fieldset id="${columnFieldSetID}"></fieldset>`);
    
            // Add each value to the html element
            columnValues.forEach(value => {
                const checkboxID = `${columnName}${value}`;
                $columnFieldSet.append($(`<input id="${checkboxID}" value="${value}" type="checkbox" class="filterCheckbox" checked/>`));
                $columnFieldSet.append($(`<label for="${checkboxID}">${value}</label><br>`));
            });
    
            $columnHtml.append($columnFieldSet);
            $filtersRow.append($columnHtml);
    
            // Make clicking the column name toggle the filters
            $(`#${columnName}btn`).click(() => {
                checkboxes = $columnFieldSet.children();
                checkboxes.each(function() {
                    this.checked = !this.checked;
                });
    
                this.onFilterUpdate();
            });
        });
    }
}

const main = function() {

    let modelFiles;
    let models = [];

    // Activate the first button and preview when a file is selected
    $("#modelDirInput").change(function(event) {
        const $filtersRow = $("#filtersRow");
        $filtersRow.empty();
        $filtersRow.append("<h3>Filter by:</h3>");

        // Load the directory
        modelFiles = Array.from(event.currentTarget.files);

        // Each directory will contain multiple files so this array will have multiple entries.
        // We'll just filter by this file to prevent duplicates
        const filterFilename = "model.ckpt.meta";

        modelFiles.forEach(file => {
            if (file.name == filterFilename) {
                models.push(buildModelDictFromFile(file));
            }
        });

        filter.models = models;

        // Now we build the checkboxes, one column at a time
        const allModelColumns = Object.keys(models[0]);

        // The timestamp and title wouldn't be very useful here
        filter.modelColumns = allModelColumns.filter(element => {
            return element !== "timestamp" && element !== "title";
        });

        filter.buildCheckboxes($filtersRow);

        // Now show the models that match the selected criteria
        $(".filterCheckbox").change(() => {filter.onFilterUpdate()});
        filter.onFilterUpdate();
    });
}

$(document).ready(main);