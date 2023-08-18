$(document).ready(function () {
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#inputQuestions')[0]);
        $('.loader').show();
        $('#result').hide();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').html('<h3><span>' + data + '</span></h3>');
            },
        });
    });

});
