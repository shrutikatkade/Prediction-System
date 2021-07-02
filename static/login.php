<?php
$con=mysqli_connect("localhost","root","","store");
// Check connection
if (mysqli_connect_errno())
  {
  echo "Failed to connect to MySQL: ".mysqli_connect_error();
  }

$mail="";
$pw="";
if($_SERVER["REQUEST_METHOD"]=="POST")
{
	
	$mail=$_REQUEST['email'];
	$pw=$_REQUEST['psw'];
}
$query = "SELECT * FROM reg WHERE Email='$mail' and Password='$pw'";
 
$result = mysqli_query($con,$query) or die(mysqli_error($connection));
if($result)
{
$count= mysqli_num_rows($result);

	if($count==1)

	{
		//successfull login
	 	$_SESSION['Email'] = $mail;
	 	echo "Welcome !";
	} 
	else
	{
		$msg = "Username or password is incorrect.";
		echo "$msg";
    }
 }   
else
{
echo "Query failed with error: " . mysqli_error($con) . "<br>";
}
mysqli_close($con);

?>
