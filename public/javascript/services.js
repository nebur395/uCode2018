angular.module('ucode18')

// 'videoFilter' service manage the exercise video filter functions of the page with the server
    .factory('videoFilter', function ($state, $http) {

        return {
            // start a video processing
            startProcess: function (callback) {
                $http({
                    method: 'POST',
                    url: 'processVideo',
                    headers: {
                        'Content-Type': 'application/json; charset=UTF-8'
                    }
                }).success(function (data) {
                    callback(data);
                }).error(function (data) {
                });
            }
        };
    });
