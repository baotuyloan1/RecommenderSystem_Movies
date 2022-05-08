<?php 

$json = file_get_contents('json_data.json');
$json_data = json_decode($json, true);
print_r($json);



?>