angular.module('ucode18')

    .controller('styleCtrl', ['$scope', 'Upload', '$timeout', '$state', function ($scope, Upload, $timeout, $state) {

        $scope.progressPercentage = 0;

        $scope.$watch('files', function () {
            $scope.upload($scope.files);
        });
        $scope.$watch('file', function () {
            if ($scope.file != null) {
                $scope.files = [$scope.file];
            }
        });
        $scope.upload = function (files) {
            if (files && files.length) {
                for (var i = 0; i < files.length; i++) {
                    var file = files[i];
                    if (!file.$error) {
                        Upload.upload({
                            url: 'https://angular-file-upload-cors-srv.appspot.com/upload',
                            data: {
                                username: $scope.username,
                                file: file
                            }
                        }).then(function (resp) {
                            $timeout(function() {
                            });
                        }, null, function (evt) {
                            $scope.progressPercentage = parseInt(100.0 *
                                evt.loaded / evt.total);
                        });
                    }
                }
            }
        };

        $scope.showOptions = false;

        $scope.$watch('progressPercentage', function () {
            if ($scope.progressPercentage === 100) {
                $scope.showOptions = true;
            }
        });

        $scope.styleImageView = "../images/style/view/original.jpg";
        $scope.dropBackgroundImage = "../images/style/drop.png";

        $scope.showClothCard = false;
        $scope.showStyleCard = false;
        $scope.showLogoCard = false;

        $scope.clothesList = [
            {id:0,src:"../images/style/clothes/camisetabase.jpg", price: 20, selected: false, model: "XX1"},
            {id:1,src:"../images/style/clothes/camisetabasenegro.jpg", price: 25, selected: false, model: "XX2"}];
        $scope.stylesList = [
            {id:0,src:"../images/style/styles/web-14.png", price: 5, selected: false, model: "XX4"},
            {id:1,src:"../images/style/styles/web-13.png", price: 5, selected: false, model: "XX3"},
            {id:2,src:"../images/style/styles/web-15.png", price: 5, selected: false, model: "XX5"},
            {id:3,src:"../images/style/styles/web-16.png", price: 5, selected: false, model: "XX6"},
            {id:4,src:"../images/style/styles/web-17.png", price: 5, selected: false, model: "XX7"},
            {id:5,src:"../images/style/styles/web-18.png", price: 5, selected: false, model: "XX8"},
            {id:6,src:"../images/style/styles/web-19.png", price: 5, selected: false, model: "XX9"},
            {id:7,src:"../images/style/styles/web-20.png", price: 5, selected: false, model: "XX10"},
            {id:8,src:"../images/style/styles/web-21.png", price: 5, selected: false, model: "XX11"},
            {id:9,src:"../images/style/styles/web-22.png", price: 5, selected: false, model: "XX12"},
            {id:10,src:"../images/style/styles/web-23.png", price: 5, selected: false, model: "XX13"},
            {id:11,src:"../images/style/styles/web-24.png", price: 5, selected: false, model: "XX14"}];
        $scope.logosList = [
            {id:0,src:"../images/style/clothes/camisetalogogrande.jpg", price: 10, selected: false, model: "XX15"},
            {id:1,src:"../images/style/clothes/camisetalogopequeno.jpg", price: 5, selected: true, model: "XX16"}];
        $scope.totalToPay = $scope.logosList[1].price;
        $scope.totalClothes = 0;
        $scope.totalStyles = 0;
        $scope.totalLogos = $scope.logosList[1].price;

        $scope.contador = 1;
        $scope.changeImage = function() {
            if ($scope.contador <= 7) {
                $scope.styleImageView = "../images/style/view/2/" + $scope.contador + ".jpg";
                $scope.contador++;
            } else {
                $scope.contador = 7;
            }
        };

        $scope.activeCloth = function(id) {
            for (i = 0; i < $scope.clothesList.length; i++) {
                if (id === $scope.clothesList[i].id && $scope.clothesList[i].selected) {
                    $scope.clothesList[i].selected = false;
                    $scope.totalClothes = 0;
                    $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
                } else if (id === $scope.clothesList[i].id && !$scope.clothesList[i].selected) {
                    $scope.clothesList[i].selected = true;
                    $scope.totalClothes = $scope.clothesList[i].price;
                    $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
                } else {
                    $scope.clothesList[i].selected = false;
                }
            }
            $scope.changeImage();
        };

        $scope.activeStyle = function(id) {
            if (id === 1) {
                $scope.activeStripes($scope.stylesList[id]);
            } else {
                for (i = 0; i < $scope.stylesList.length; i++) {
                    if (id === $scope.stylesList[i].id && $scope.stylesList[i].selected) {
                        $scope.stylesList[i].selected = false;
                        $scope.totalStyles -= $scope.stylesList[i].price;
                        $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
                    } else if (id === $scope.stylesList[i].id && !$scope.stylesList[i].selected) {
                        $scope.stylesList[i].selected = true;
                        $scope.totalStyles += $scope.stylesList[i].price;
                        $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
                    } else if ($scope.stylesList[i].id !== 1 && $scope.stylesList[i].selected) {
                        $scope.stylesList[i].selected = false;
                        $scope.totalStyles -= $scope.stylesList[i].price;
                    } else if ($scope.stylesList[i].id !== 1) {
                        $scope.stylesList[i].selected = false;
                    }
                }
            }
            $scope.changeImage();
        };

        $scope.activeStripes = function(stripe) {
            if (stripe.selected) {
                stripe.selected = false;
                $scope.totalStyles -= stripe.price;
                $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
            } else {
                stripe.selected = true;
                $scope.totalStyles += stripe.price;
                $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
            }
        };

        $scope.activeLogo = function(id) {
            for (i = 0; i < $scope.logosList.length; i++) {
                if (id === $scope.logosList[i].id && $scope.logosList[i].selected) {
                    $scope.totalLogos = 0;
                    $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
                } else if (id === $scope.logosList[i].id && !$scope.logosList[i].selected) {
                    $scope.logosList[i].selected = true;
                    $scope.totalLogos = $scope.logosList[i].price;
                    $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
                } else {
                    $scope.logosList[i].selected = false;
                }
            }
            $scope.changeImage();
        };

        $scope.openCard = function(card) {
            switch (card) {
                case 1:
                    $scope.showClothCard = !$scope.showClothCard;
                    $scope.showStyleCard = false;
                    $scope.showLogoCard = false;
                    break;
                case 2:
                    $scope.showClothCard = false;
                    $scope.showStyleCard = !$scope.showStyleCard;
                    $scope.showLogoCard = false;
                    break;
                case 3:
                    $scope.showClothCard = false;
                    $scope.showStyleCard = false;
                    $scope.showLogoCard = !$scope.showLogoCard;
                    break;
            }
        };

        $scope.goStarter = function () {
            $state.go('starter');
        };


    }]);
