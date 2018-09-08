# T. A. Davis
# myfind.pl
#
# This program mimics the unix/linux find command, but with limited arguments:
#   -name string    searches for files matching name (wild cards permitted)
#   -ls             long listing of files
#   -pwd            fully qualified path name listed for files
#   -grep string    finds matching lines in files and prints name and line number
#
# This program makes use of perl code from the following sources:
#   http://docstore.mik.ua/orelly/perl/cookbook/ch06_10.htm
#   http://docstore.mik.ua/orelly/perl/sysadmin/ch02_03.htm

#! /usr/local/bin/perl

use strict;

# global variables for arguments
my $myname = 0;
my $myls   = 0;
my $mygrep = 0;
my $mypwd  = 0;
my $myfilesearch = "";
my $mygrepsearch = "";


# This subroutine can be found at http://docstore.mik.ua/orelly/perl/sysadmin/ch02_03.htm
sub glob2pat {

   my $globstr = shift;

   my %patmap = (
        '*' => '.*',
        '?' => '.',
        '[' => '[',
        ']' => ']',
   );

   $globstr =~ s{(.)} { $patmap{$1} || "\Q$1" }ge;

   return '^' . $globstr . '$';
}


# The following subroutine is derived from http://docstore.mik.ua/orelly/perl/cookbook/ch06_10.htm.
# It takes the name of a directory and recursively scans down the filesystem from that point.

use Cwd; # module for finding the current working directory

sub ScanDirectory {

   # get the start directory from the passed argument
   my ($workdir) = shift;

   # keep track of where we began
   my ($startdir) = &cwd;

   # change directory to start directory
   chdir($workdir) or die "Unable to enter dir $workdir:$!\n";

   # read directory and store filenames in names arry
   opendir(DIR, ".") or die "Unable to open $workdir:$!\n";
   my @names = readdir(DIR) or die "Unable to read $workdir:$!\n";
   closedir(DIR);

   # process -pwd argument
   my $pwd = "";

   if ($mypwd) {

      # create prefix for filenames
      $pwd = &cwd . "/";
   }

   foreach my $name (@names){

      next if ($name eq "."); 
      next if ($name eq "..");

      # if directory, recurse
      if (-d $name) {

         &ScanDirectory($name);
         next;
      }

      # process -name argument
      if ($myname) {


         next unless ($name =~ m/$myfilesearch/);
      }

      # process -ls argument
      if ($myls) {

            my $newname = $pwd . $name;
            system ("ls -l $newname");
      }

      # process -grep argument
      elsif ($mygrep) {

         my $linenum = 0;
         my $myfile;

         open ($myfile, $name);

         # find lines in file matching grep pattern
         while (<$myfile>) {

            $linenum++;

            print $pwd . $name, ": $linenum: $_" if /$mygrepsearch/;
         }

         close ($name);
      }

      # just print filenames if no arguments
      else {

         print $pwd . $name, "\n";
      }
   }

   chdir($startdir) or die "Unable to change to dir $startdir:$!\n";
}

# main program

# initialize start directory to current directory
my $mystartdir = ".";

# re-set start directory if any arguments given
if (@ARGV > 0)  { 

   $mystartdir = shift;
}

# process arguments
while (@ARGV) {

   my $myarg = shift;

   if ($myarg eq "-ls") {

      $myls = 1;
   }

   elsif ($myarg eq "-pwd") {

      $mypwd = 1;
   }

   elsif ($myarg eq "-name") { 

      $myname = 1;
      $myfilesearch = shift;

      # strip off quotation marks and convert to regular expression
      $myfilesearch =~ s/"//g;
      $myfilesearch = glob2pat($myfilesearch);
   }

   elsif ($myarg eq "-grep") {

      $mygrep = 1;
      $mygrepsearch = shift;

      # strip off quotation marks
      $mygrepsearch =~ s/"//g;
   }
}

&ScanDirectory($mystartdir);

